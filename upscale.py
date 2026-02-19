#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["click", "Pillow"]
# ///
# ABOUTME: Wrapper script for iris.c super-resolution upscaling with smart scaling and evolution
# ABOUTME: Adds dimension inference, percentage scaling, iterative evolution, and persistent reference images

import hashlib
import random
import shlex
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import click
from PIL import Image

from iris_ffi import (
    IrisContext, IrisParams, IRIS_SCHEDULE_LINEAR, IRIS_SCHEDULE_POWER,
    display_png_standalone,
)

DEFAULT_PROMPT = "Create an exact copy of the input image."


def align16(n):
    """Round down to nearest multiple of 16."""
    return (n // 16) * 16


def get_version():
    """Get git short SHA of the upscale repo, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def sha256_file(path):
    """Return hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_command_string(args, input_paths, seed, width, height):
    """Build the upscale command string that reproduces this exact generation.

    Includes generation parameters only (inputs, dimensions, prompt, steps,
    guidance, seed, model, schedule). Excludes batch/display options.
    """
    parts = ["upscale"]

    for p in input_paths:
        parts.append(f"-i {shlex.quote(str(p))}")

    if width is not None:
        parts.append(f"-W {width}")
    if height is not None:
        parts.append(f"-H {height}")

    if args.prompt != DEFAULT_PROMPT:
        parts.append(f"-p {shlex.quote(args.prompt)}")

    if args.steps is not None:
        parts.append(f"-s {args.steps}")
    if args.guidance is not None:
        parts.append(f"-g {args.guidance}")

    parts.append(f"-S {seed}")

    if args.base:
        parts.append("--base")
    if args.nine_b:
        parts.append("--9b")

    if args.linear:
        parts.append("--linear")
    elif args.power:
        parts.append("--power")
    if args.power_alpha is not None:
        parts.append(f"--power-alpha {args.power_alpha}")

    return " ".join(parts)


def make_png_text_chunk(keyword, text):
    """Build a PNG tEXt chunk as raw bytes.

    Format per PNG spec: [4-byte length][tEXt][keyword\\0text][4-byte CRC32].
    Mirrors write_png_text_chunk() in iris_image.c.
    """
    payload = keyword.encode() + b"\x00" + text.encode()
    chunk_type = b"tEXt"
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def inject_png_metadata(png_path, metadata):
    """Inject tEXt metadata chunks into an existing PNG file.

    Reads the file, finds the insertion point before the first IDAT chunk,
    splices in new tEXt chunks, and writes back. No pixel decode/encode.
    """
    raw = Path(png_path).read_bytes()
    new_chunks = b"".join(make_png_text_chunk(k, v) for k, v in metadata.items())

    # Scan chunks to find first IDAT (insertion point)
    pos = 8  # skip PNG signature
    while pos < len(raw):
        chunk_len = struct.unpack(">I", raw[pos:pos + 4])[0]
        chunk_type = raw[pos + 4:pos + 8]
        if chunk_type == b"IDAT":
            break
        pos += 12 + chunk_len  # 4 len + 4 type + data + 4 crc

    Path(png_path).write_bytes(raw[:pos] + new_chunks + raw[pos:])


def get_dimensions(args, input_path, max_area):
    """Compute output dimensions from args and optional input image.

    Returns (in_w, in_h, out_w, out_h). in_w/in_h are None when no input image.
    out_w/out_h are None when iris should auto-detect.
    Prints a warning to stderr if the output area exceeds max_area.
    """
    if input_path:
        img = Image.open(input_path)
        in_w, in_h = img.size
        img.close()
    else:
        in_w, in_h = None, None

    width, height = None, None

    if args.scale is not None:
        factor = args.scale / 100
        width = align16(round(in_w * factor))
        height = align16(round(in_h * factor))
    elif args.width is not None or args.height is not None:
        if args.width is not None and args.height is not None:
            width, height = args.width, args.height
        elif args.width is not None:
            width = args.width
            if in_h:
                height = align16(round(width * in_h / in_w))
        else:
            height = args.height
            if in_w:
                width = align16(round(height * in_w / in_h))

    # Area warning
    check_w = width if width is not None else (in_w or 256)
    check_h = height if height is not None else (in_h or 256)
    area = check_w * check_h
    if area > max_area:
        megapixels = area / 1_000_000
        print(
            f"Warning: output area {check_w}x{check_h} = {megapixels:.1f} MP "
            f"exceeds --max-area {max_area} ({max_area / 1_000_000:.1f} MP)",
            file=sys.stderr,
        )

    return in_w, in_h, width, height


def resolve_output_path(output_str, input_path=None):
    """Resolve the output path from the optional output argument.

    Rules:
    - None (omitted): auto-number using input stem (or "generated") in current directory
    - Ends with '/': treat as directory (created if needed), auto-number using input stem
    - Ends with '.png': use as-is, overwrite on subsequent runs
    - Otherwise: treat as base name, auto-number with that name

    Returns (resolved_path, auto_numbered).
    """
    stem = input_path.stem if input_path else "generated"
    if output_str is None:
        directory = Path(".")
        base_name = stem
    elif output_str.endswith("/"):
        directory = Path(output_str)
        base_name = stem
    elif output_str.endswith(".png"):
        return Path(output_str), False
    else:
        p = Path(output_str)
        directory = p.parent
        base_name = p.name

    # Auto-create directory if needed
    directory.mkdir(parents=True, exist_ok=True)

    # Find next free number (check both plain and evolution-numbered variants)
    for n in range(100000):
        candidate = directory / f"{base_name}-{n:05d}.png"
        if not candidate.exists() and not list(directory.glob(f"{base_name}-{n:05d}_*.png")):
            return candidate, True

    print(f"Error: could not find a free filename for {base_name}-* in {directory}", file=sys.stderr)
    sys.exit(1)


def make_output_path(base_output, iteration, total):
    """Return the output path for this iteration.

    Single iteration: use base_output directly.
    Multiple iterations: use {stem}_{NNN}.png numbering.
    """
    base_output = base_output.with_suffix(".png")
    if total == 1:
        return base_output
    return base_output.with_stem(f"{base_output.stem}_{iteration:03d}")


def build_params(args, width, height, seed):
    """Build an IrisParams from parsed arguments and per-iteration overrides."""
    params = IrisParams.default()
    params.width = width if width is not None else 256
    params.height = height if height is not None else 256
    params.seed = seed if seed is not None else -1
    if args.steps is not None:
        params.num_steps = args.steps
    if args.guidance is not None:
        params.guidance = args.guidance
    if args.linear:
        params.schedule = IRIS_SCHEDULE_LINEAR
    elif args.power:
        params.schedule = IRIS_SCHEDULE_POWER
    if args.power_alpha is not None:
        params.power_alpha = args.power_alpha
    return params


EXAMPLES = """\b
Examples:
  upscale -i input.png
  upscale -i input.png --scale 200
  upscale -i input.png -W 1024
  upscale -i input.png --base
  upscale -i input.png --count 3 --evolve 2
  upscale -p "a cat" -W 512 -H 512
  upscale info output.png
  upscale show output.png"""


@click.group(invoke_without_command=True, epilog=EXAMPLES)
@click.option("-o", "output", default=None,
              help="Output path (.png = overwrite, no ext = auto-number, trailing / = directory)")
@click.option("--base", is_flag=True, help="Use base model (higher quality, ~25x slower)")
@click.option("--9b", "nine_b", is_flag=True,
              help="Use 9B model (larger, non-commercial). Download with: HF_TOKEN=... just download-9b")
@click.option("-i", "images", multiple=True, type=click.Path(path_type=Path),
              help="Input image (repeatable). First is primary (evolves), rest are persistent refs.")
@click.option("-W", "--width", type=int, default=None, help="Output width (default: iris auto-detect)")
@click.option("-H", "--height", type=int, default=None, help="Output height (default: iris auto-detect)")
@click.option("--scale", type=float, default=None,
              help="Scale percentage (e.g. 200 = 2x). Mutually exclusive with -W/-H.")
@click.option("--evolve", type=int, default=1, help="Number of evolution iterations (default: 1)")
@click.option("--count", type=int, default=1,
              help="Generate N images with different seeds, model loaded once (default: 1)")
@click.option("--max-area", type=int, default=1_048_576,
              help="Pixel area warning threshold (default: 1048576 = 1024x1024)")
@click.option("-p", "--prompt", default=DEFAULT_PROMPT,
              help=f'Text prompt (default: "{DEFAULT_PROMPT}")')
@click.option("-s", "--steps", type=int, default=None,
              help="Sampling steps (default: auto, 4 distilled / 50 base)")
@click.option("-g", "--guidance", type=float, default=None,
              help="CFG guidance scale (default: auto, 1.0 distilled / 4.0 base)")
@click.option("-S", "--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--linear", is_flag=True,
              help="Use linear timestep schedule (faster preview with fewer steps)")
@click.option("--power", is_flag=True, help="Use power curve timestep schedule")
@click.option("--power-alpha", type=float, default=None, help="Power schedule exponent (default: 2.0)")
@click.option("--show", is_flag=True, help="Display final image in terminal (Kitty/Ghostty/iTerm2/WezTerm)")
@click.option("--show-steps", "show_steps", is_flag=True,
              help="Display each denoising step in terminal (slower)")
@click.pass_context
def cli(ctx, **kwargs):
    """Upscale or generate images using Flux.2 Klein super-resolution."""
    if ctx.invoked_subcommand is not None:
        return

    args = SimpleNamespace(**kwargs)
    script_dir = Path(__file__).resolve().parent
    version = get_version()

    # Split -i images: first is primary (evolves), rest are persistent refs
    primary_input = args.images[0] if args.images else None
    persistent_refs = args.images[1:] if len(args.images) > 1 else []

    # Select model directory based on --9b and --base flags
    size = "9b" if args.nine_b else "4b"
    if args.base:
        model_dir = script_dir / f"flux-klein-{size}-base"
    else:
        model_dir = script_dir / f"flux-klein-{size}"

    # --- Validation ---

    if args.scale is not None and (args.width is not None or args.height is not None):
        raise click.UsageError("--scale and -W/-H are mutually exclusive")

    if args.scale is not None and args.scale <= 0:
        raise click.UsageError("--scale must be positive")

    if args.evolve < 1:
        raise click.UsageError("--evolve must be >= 1")

    if args.count < 1:
        raise click.UsageError("--count must be >= 1")

    if primary_input is None:
        # Text-to-image or ref-only mode
        if args.scale is not None:
            raise click.UsageError("--scale requires an input image (-i)")
        if (args.width is not None) != (args.height is not None):
            raise click.UsageError("Without an input image, both -W and -H are required "
                                   "(can't infer aspect ratio)")
        if args.prompt == DEFAULT_PROMPT:
            raise click.UsageError("-p/--prompt is required when no input image is given")
    else:
        if not primary_input.exists():
            raise click.BadParameter(f"Input file not found: {primary_input}", param_hint="'-i'")

    for ref in persistent_refs:
        if not ref.exists():
            raise click.BadParameter(f"Reference image not found: {ref}", param_hint="'-i'")

    # Resolve output path (auto-number, directory creation, etc.)
    output_path, auto_numbered = resolve_output_path(args.output, primary_input)

    # For explicit .png paths, validate parent directory exists
    if not auto_numbered:
        output_dir = output_path.parent
        if str(output_dir) != "." and not output_dir.exists():
            raise click.BadParameter(f"Output directory not found: {output_dir}", param_hint="'-o'")

    if not IrisContext.available():
        raise click.ClickException(
            "libiris.dylib not found. Run 'just' first to build iris.c and the shared library")

    if not model_dir.exists():
        model_name = "base model" if args.base else "model"
        raise click.ClickException(
            f"{model_name.capitalize()} not found at {model_dir}. "
            "Run 'just' first to download the models")

    # --- Dimension calculation (iteration 1 only) ---

    in_w, in_h, width, height = get_dimensions(args, primary_input, args.max_area)

    # --- Generation loop (count x evolve) ---

    base_seed = args.seed
    count = args.count
    evolve = args.evolve
    total_images = count * evolve

    try:
        t0 = time.monotonic()
        print(f"Loading model from {model_dir}...")
        iris_ctx = IrisContext(model_dir)
        print(f"Model loaded ({time.monotonic() - t0:.1f}s)")

        if args.show_steps:
            iris_ctx.enable_show_steps()

        show = args.show or args.show_steps

        # Pre-encode text and image for multi-seed generation (--count > 1).
        # Saves ~1.5-15s per additional seed by avoiding redundant Qwen3 and VAE work.
        # Only when there's at least one image — pure txt2img lets iris handle encoding.
        use_precomputed = count > 1 and (primary_input is not None or persistent_refs)
        text_emb = text_emb_uncond = first_img_cache = None
        persistent_ref_latents = []  # pre-encoded persistent ref images

        if use_precomputed:
            text_emb, text_seq = iris_ctx.encode_text(args.prompt)
            if not iris_ctx.is_distilled():
                text_emb_uncond, text_seq_uncond = iris_ctx.encode_text("")
            else:
                text_seq_uncond = 0
            iris_ctx.release_text_encoder()

            # Pre-encode persistent reference images (reused across all seeds/evolve)
            for ref_path in persistent_refs:
                persistent_ref_latents.append(iris_ctx.encode_image(ref_path))

        for seed_idx in range(count):
            # Determine seed for this count iteration
            if seed_idx == 0 and base_seed is not None:
                seed = base_seed
            else:
                seed = random.randint(0, 2**63 - 1)

            # For count > 1, each seed gets its own auto-numbered base
            if seed_idx == 0:
                seed_output = output_path
            else:
                seed_output, _ = resolve_output_path(args.output, primary_input)

            evolving_input = primary_input

            for evo_idx in range(1, evolve + 1):
                out_path = make_output_path(seed_output, evo_idx, evolve)

                # Only pass dimensions on first evolution iteration; subsequent ones let iris auto-detect
                iter_w = width if evo_idx == 1 else None
                iter_h = height if evo_idx == 1 else None

                params = build_params(args, iter_w, iter_h, seed)

                if iter_w and in_w:
                    dim_str = f" ({in_w}x{in_h} -> {iter_w}x{iter_h})"
                elif iter_w:
                    dim_str = f" ({iter_w}x{iter_h})"
                elif in_w:
                    dim_str = f" ({in_w}x{in_h})"
                else:
                    dim_str = ""

                label_parts = []
                if count > 1:
                    label_parts.append(f"seed {seed_idx + 1}/{count}")
                if evolve > 1:
                    label_parts.append(f"evolve {evo_idx}/{evolve}")

                # Build input description: primary + any persistent refs
                input_parts = []
                if evolving_input:
                    input_parts.append(str(evolving_input))
                input_parts.extend(str(r) for r in persistent_refs)
                inputs = " + ".join(input_parts) if input_parts else "(text-to-image)"

                if label_parts:
                    label = f"[{', '.join(label_parts)}]"
                    print(f"{label}: {inputs} -> {out_path}{dim_str} (seed={seed})")
                else:
                    action = "Upscaling" if primary_input else "Generating"
                    print(f"{action}: {inputs} -> {out_path}{dim_str} (seed={seed})")

                gen_t0 = time.monotonic()

                if use_precomputed:
                    if evolving_input:
                        # Cache primary image latent for first evolve step (shared across seeds)
                        if evo_idx == 1:
                            if first_img_cache is None:
                                first_img_cache = iris_ctx.encode_image(evolving_input)
                            primary_lat = first_img_cache
                        else:
                            primary_lat = iris_ctx.encode_image(evolving_input)
                        all_ref_latents = [primary_lat] + persistent_ref_latents
                    else:
                        all_ref_latents = list(persistent_ref_latents)

                    iris_ctx.multiref_precomputed(
                        text_emb, text_seq, text_emb_uncond, text_seq_uncond,
                        all_ref_latents, out_path, params,
                    )

                    # Free per-evolve primary latents (not the cached first one)
                    if evolving_input and evo_idx > 1:
                        iris_ctx.free_encoded(primary_lat[0])
                else:
                    input_images = []
                    if evolving_input:
                        input_images.append(evolving_input)
                    input_images.extend(persistent_refs)
                    iris_ctx.multiref(args.prompt, input_images, out_path, params)

                gen_elapsed = time.monotonic() - gen_t0
                print(f"Saved {out_path} ({gen_elapsed:.1f}s)")

                # Inject reproducibility metadata into the saved PNG
                gen_inputs = []
                if evolving_input:
                    gen_inputs.append(evolving_input)
                gen_inputs.extend(persistent_refs)

                meta = {"upscale:command": build_command_string(
                    args, gen_inputs, seed, iter_w, iter_h,
                )}
                if gen_inputs:
                    meta["upscale:input_sha256"] = ",".join(
                        sha256_file(p) for p in gen_inputs
                    )
                meta["upscale:version"] = version
                inject_png_metadata(out_path, meta)

                if show:
                    iris_ctx.display_png(out_path)

                evolving_input = out_path

        # Free pre-encoded data
        if text_emb:
            iris_ctx.free_encoded(text_emb)
        if text_emb_uncond:
            iris_ctx.free_encoded(text_emb_uncond)
        if first_img_cache:
            iris_ctx.free_encoded(first_img_cache[0])
        for ref_lat in persistent_ref_latents:
            iris_ctx.free_encoded(ref_lat[0])

        iris_ctx.close()

        total_elapsed = time.monotonic() - t0
        if total_images > 1:
            print(f"Done: {total_images} images generated ({total_elapsed:.1f}s total)")
        else:
            print(f"Done: {out_path} ({total_elapsed:.1f}s total)")

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def info(path):
    """Display embedded metadata from a generated image."""
    if path.suffix.lower() != ".png":
        raise click.BadParameter(f"Not a PNG file: {path}", param_hint="'PATH'")

    img = Image.open(path)
    text = getattr(img, "text", {})
    img.close()

    if not text:
        print(f"No text metadata found in {path}")
        return

    # Print upscale-specific chunks first, then iris, then others
    upscale_keys = [k for k in text if k.startswith("upscale:")]
    iris_keys = [k for k in text if k.startswith("iris:") or k == "Software"]
    other_keys = [k for k in text if k not in upscale_keys and k not in iris_keys]

    for key in upscale_keys + iris_keys + other_keys:
        print(f"{key}: {text[key]}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def show(path):
    """Display an image in the terminal (Kitty/Ghostty/iTerm2/WezTerm)."""
    if not IrisContext.available():
        raise click.ClickException(
            "libiris.dylib not found. Run 'just' first to build iris.c and the shared library")
    display_png_standalone(path)


if __name__ == "__main__":
    cli()
