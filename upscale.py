#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow"]
# ///
# ABOUTME: Wrapper script for iris.c super-resolution upscaling with smart scaling and evolution
# ABOUTME: Adds dimension inference, percentage scaling, iterative evolution, and persistent reference images

import argparse
import random
import sys
import time
from pathlib import Path

from PIL import Image

from iris_ffi import IrisContext, IrisParams, IRIS_SCHEDULE_LINEAR, IRIS_SCHEDULE_POWER

DEFAULT_PROMPT = "Create an exact copy of the input image."


def align16(n):
    """Round down to nearest multiple of 16."""
    return (n // 16) * 16


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


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Upscale or generate images using Flux.2 Klein super-resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale: auto-numbered output (input-00000.png)
  %(prog)s -i input.png

  # Output to directory (created if needed): results/input-00000.png
  %(prog)s -i input.png -o results/

  # Custom base name: upscaled-00000.png
  %(prog)s -i input.png -o upscaled

  # Explicit .png: always overwrites
  %(prog)s -i input.png -o output.png

  # Scale to 200%% (2x), dimensions aligned to 16px
  %(prog)s -i input.png --scale 200

  # Evolve 3 iterations with a persistent reference image
  %(prog)s -i input.png -i ref.png --evolve 3 -p "blend with reference"

  # Generate 3 variations with different seeds, model loaded once
  %(prog)s -i input.png --count 3

  # Text-to-image (no input image)
  %(prog)s -p "a cat sitting on a rainbow" -W 512 -H 512

  # Multi-reference generation
  %(prog)s -i car.png -i beach.png -p "sports car on beach" -W 512 -H 512
""",
    )

    parser.add_argument(
        "-o", dest="output", default=None,
        help="Output path (.png = overwrite, no ext = auto-number, trailing / = directory)",
    )
    parser.add_argument(
        "--base", action="store_true",
        help="Use base model (higher quality, ~25x slower)",
    )
    parser.add_argument(
        "--9b", dest="nine_b", action="store_true",
        help="Use 9B model (larger, non-commercial). Download with: HF_TOKEN=... just download-9b",
    )
    parser.add_argument(
        "-i",
        dest="images",
        type=Path,
        action="append",
        default=[],
        help="Input image (repeatable). First is primary (evolves), rest are persistent refs.",
    )
    parser.add_argument(
        "-W", "--width", type=int, default=None, help="Output width (default: iris auto-detect)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=None, help="Output height (default: iris auto-detect)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale percentage (e.g. 200 = 2x). Mutually exclusive with -W/-H.",
    )
    parser.add_argument(
        "--evolve",
        type=int,
        default=1,
        help="Number of evolution iterations (default: 1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Generate N images with different seeds, model loaded once (default: 1)",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=1_048_576,
        help="Pixel area warning threshold (default: 1048576 = 1024x1024)",
    )

    iris_group = parser.add_argument_group("Generation options")
    iris_group.add_argument(
        "-p", "--prompt", default=DEFAULT_PROMPT,
        help=f'Text prompt (default: "{DEFAULT_PROMPT}")',
    )
    iris_group.add_argument(
        "-s", "--steps", type=int, default=None,
        help="Sampling steps (default: auto, 4 distilled / 50 base)",
    )
    iris_group.add_argument(
        "-g", "--guidance", type=float, default=None,
        help="CFG guidance scale (default: auto, 1.0 distilled / 4.0 base)",
    )
    iris_group.add_argument(
        "-S", "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    iris_group.add_argument(
        "--linear", action="store_true",
        help="Use linear timestep schedule (faster preview with fewer steps)",
    )
    iris_group.add_argument(
        "--power", action="store_true",
        help="Use power curve timestep schedule",
    )
    iris_group.add_argument(
        "--power-alpha", type=float, default=None,
        help="Power schedule exponent (default: 2.0)",
    )
    iris_group.add_argument(
        "--show", action="store_true",
        help="Display final image in terminal (Kitty/Ghostty/iTerm2/WezTerm)",
    )
    iris_group.add_argument(
        "--show-steps", action="store_true",
        help="Display each denoising step in terminal (slower)",
    )
    args = parser.parse_args()

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
        parser.error("--scale and -W/-H are mutually exclusive")

    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")

    if args.evolve < 1:
        parser.error("--evolve must be >= 1")

    if args.count < 1:
        parser.error("--count must be >= 1")

    if primary_input is None:
        # Text-to-image or ref-only mode
        if args.scale is not None:
            parser.error("--scale requires an input image (-i)")
        if (args.width is not None) != (args.height is not None):
            parser.error("Without an input image, both -W and -H are required (can't infer aspect ratio)")
        if args.prompt == DEFAULT_PROMPT:
            parser.error("-p/--prompt is required when no input image is given")
    else:
        if not primary_input.exists():
            print(f"Error: Input file not found: {primary_input}", file=sys.stderr)
            sys.exit(1)

    for ref in persistent_refs:
        if not ref.exists():
            print(f"Error: Reference image not found: {ref}", file=sys.stderr)
            sys.exit(1)

    # Resolve output path (auto-number, directory creation, etc.)
    output_path, auto_numbered = resolve_output_path(args.output, primary_input)

    # For explicit .png paths, validate parent directory exists
    if not auto_numbered:
        output_dir = output_path.parent
        if str(output_dir) != "." and not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
            sys.exit(1)

    if not IrisContext.available():
        print("Error: libiris.dylib not found", file=sys.stderr)
        print("Run 'just' first to build iris.c and the shared library", file=sys.stderr)
        sys.exit(1)

    if not model_dir.exists():
        model_name = "base model" if args.base else "model"
        print(f"Error: {model_name.capitalize()} not found at {model_dir}", file=sys.stderr)
        print("Run 'just' first to download the models", file=sys.stderr)
        sys.exit(1)

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
        ctx = IrisContext(model_dir)
        print(f"Model loaded ({time.monotonic() - t0:.1f}s)")

        if args.show_steps:
            ctx.enable_show_steps()

        show = args.show or args.show_steps

        # Pre-encode text and image for multi-seed generation (--count > 1).
        # Saves ~1.5-15s per additional seed by avoiding redundant Qwen3 and VAE work.
        # Only when there's at least one image — pure txt2img lets iris handle encoding.
        use_precomputed = count > 1 and (primary_input is not None or persistent_refs)
        text_emb = text_emb_uncond = first_img_cache = None
        persistent_ref_latents = []  # pre-encoded persistent ref images

        if use_precomputed:
            text_emb, text_seq = ctx.encode_text(args.prompt)
            if not ctx.is_distilled():
                text_emb_uncond, text_seq_uncond = ctx.encode_text("")
            else:
                text_seq_uncond = 0
            ctx.release_text_encoder()

            # Pre-encode persistent reference images (reused across all seeds/evolve)
            for ref_path in persistent_refs:
                persistent_ref_latents.append(ctx.encode_image(ref_path))

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
                                first_img_cache = ctx.encode_image(evolving_input)
                            primary_lat = first_img_cache
                        else:
                            primary_lat = ctx.encode_image(evolving_input)
                        all_ref_latents = [primary_lat] + persistent_ref_latents
                    else:
                        all_ref_latents = list(persistent_ref_latents)

                    ctx.multiref_precomputed(
                        text_emb, text_seq, text_emb_uncond, text_seq_uncond,
                        all_ref_latents, out_path, params,
                    )

                    # Free per-evolve primary latents (not the cached first one)
                    if evolving_input and evo_idx > 1:
                        ctx.free_encoded(primary_lat[0])
                else:
                    input_images = []
                    if evolving_input:
                        input_images.append(evolving_input)
                    input_images.extend(persistent_refs)
                    ctx.multiref(args.prompt, input_images, out_path, params)

                gen_elapsed = time.monotonic() - gen_t0
                print(f"Saved {out_path} ({gen_elapsed:.1f}s)")

                if show:
                    ctx.display_png(out_path)

                evolving_input = out_path

        # Free pre-encoded data
        if text_emb:
            ctx.free_encoded(text_emb)
        if text_emb_uncond:
            ctx.free_encoded(text_emb_uncond)
        if first_img_cache:
            ctx.free_encoded(first_img_cache[0])
        for ref_lat in persistent_ref_latents:
            ctx.free_encoded(ref_lat[0])

        ctx.close()

        total_elapsed = time.monotonic() - t0
        if total_images > 1:
            print(f"Done: {total_images} images generated ({total_elapsed:.1f}s total)")
        else:
            print(f"Done: {out_path} ({total_elapsed:.1f}s total)")

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
