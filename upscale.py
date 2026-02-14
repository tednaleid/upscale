#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow"]
# ///
# ABOUTME: Wrapper script for iris.c super-resolution upscaling with smart scaling and evolution
# ABOUTME: Adds dimension inference, percentage scaling, iterative evolution, and persistent reference images

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image


def align16(n):
    """Round down to nearest multiple of 16."""
    return (n // 16) * 16


def get_dimensions(args, input_path, max_area):
    """Compute output dimensions from args and input image.

    Returns (in_w, in_h, out_w, out_h). out_w/out_h are None when iris should auto-detect.
    Prints a warning to stderr if the output area exceeds max_area.
    """
    img = Image.open(input_path)
    in_w, in_h = img.size
    img.close()

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
            height = align16(round(width * in_h / in_w))
        else:
            height = args.height
            width = align16(round(height * in_w / in_h))

    # Area warning
    check_w = width if width is not None else in_w
    check_h = height if height is not None else in_h
    area = check_w * check_h
    if area > max_area:
        megapixels = area / 1_000_000
        print(
            f"Warning: output area {check_w}x{check_h} = {megapixels:.1f} MP "
            f"exceeds --max-area {max_area} ({max_area / 1_000_000:.1f} MP)",
            file=sys.stderr,
        )

    return in_w, in_h, width, height


def resolve_output_path(output_str, input_path):
    """Resolve the output path from the optional output argument.

    Rules:
    - None (omitted): auto-number using input stem in current directory
    - Ends with '/': treat as directory (created if needed), auto-number using input stem
    - Ends with '.png': use as-is, overwrite on subsequent runs
    - Otherwise: treat as base name, auto-number with that name

    Returns (resolved_path, auto_numbered).
    """
    if output_str is None:
        directory = Path(".")
        base_name = input_path.stem
    elif output_str.endswith("/"):
        directory = Path(output_str)
        base_name = input_path.stem
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


def run_iris(iris_bin, model_dir, input_images, output_path, width, height, extra_args):
    """Build and run a single iris command. Returns the CompletedProcess."""
    cmd = [str(iris_bin), "-d", str(model_dir)]

    for img in input_images:
        cmd.extend(["-i", str(img)])

    if width is not None and height is not None:
        cmd.extend(["-W", str(width), "-H", str(height)])

    cmd.extend(["-o", str(output_path)])
    cmd.extend(extra_args)

    print(f"$ {shlex.join(cmd)}")
    return subprocess.run(cmd)


def main():
    script_dir = Path(__file__).resolve().parent
    iris_bin = script_dir / "iris.c" / "iris"

    parser = argparse.ArgumentParser(
        description="Upscale an image using Flux.2 Klein 4B super-resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-number output: creates input-00000.png (or next free number)
  %(prog)s input.png

  # Output to directory (created if needed): results/input-00000.png
  %(prog)s input.png results/

  # Custom base name: upscaled-00000.png
  %(prog)s input.png upscaled

  # Explicit .png: always overwrites
  %(prog)s input.png output.png

  # Scale to 200%% (2x), dimensions aligned to 16px
  %(prog)s input.png --scale 200

  # Evolve 3 iterations with a persistent reference image
  %(prog)s input.png -i ref.png --evolve 3 -p "blend with reference"

  # Show in terminal, verbose, 8 steps
  %(prog)s input.png --show -v -s 8
""",
    )

    parser.add_argument("input", type=Path, help="Path to input image")
    parser.add_argument(
        "output", nargs="?", default=None,
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
        dest="ref_images",
        type=Path,
        action="append",
        default=[],
        help="Additional reference image (repeatable, passed to every iteration)",
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
        "--max-area",
        type=int,
        default=1_048_576,
        help="Pixel area warning threshold (default: 1048576 = 1024x1024)",
    )

    iris_group = parser.add_argument_group(
        "Common iris options (passed through to iris)"
    )
    iris_group.add_argument(
        "-p", "--prompt", default="Create an exact copy of the input image.",
        help='Text prompt (default: "Create an exact copy of the input image.")',
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
        "-v", "--verbose", action="store_true",
        help="Show detailed output",
    )
    iris_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Silent mode, no output",
    )
    iris_group.add_argument(
        "--show", action="store_true",
        help="Display image in terminal (Kitty/Ghostty/iTerm2)",
    )
    iris_group.add_argument(
        "--show-steps", action="store_true",
        help="Display each denoising step (slower)",
    )

    args, extra_args = parser.parse_known_args()

    # Select model directory based on --9b and --base flags
    size = "9b" if args.nine_b else "4b"
    if args.base:
        model_dir = script_dir / f"flux-klein-{size}-base"
    else:
        model_dir = script_dir / f"flux-klein-{size}"

    # Rebuild known iris flags into the passthrough list
    iris_passthrough = ["-p", args.prompt]
    if args.steps is not None:
        iris_passthrough.extend(["-s", str(args.steps)])
    if args.guidance is not None:
        iris_passthrough.extend(["-g", str(args.guidance)])
    if args.seed is not None:
        iris_passthrough.extend(["-S", str(args.seed)])
    if args.linear:
        iris_passthrough.append("--linear")
    if args.power:
        iris_passthrough.append("--power")
    if args.power_alpha is not None:
        iris_passthrough.extend(["--power-alpha", str(args.power_alpha)])
    if args.verbose:
        iris_passthrough.append("-v")
    if args.quiet:
        iris_passthrough.append("-q")
    if args.show:
        iris_passthrough.append("--show")
    if args.show_steps:
        iris_passthrough.append("--show-steps")
    iris_passthrough.extend(extra_args)

    # --- Validation ---

    if args.scale is not None and (args.width is not None or args.height is not None):
        parser.error("--scale and -W/-H are mutually exclusive")

    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")

    if args.evolve < 1:
        parser.error("--evolve must be >= 1")

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    for ref in args.ref_images:
        if not ref.exists():
            print(f"Error: Reference image not found: {ref}", file=sys.stderr)
            sys.exit(1)

    # Resolve output path (auto-number, directory creation, etc.)
    output_path, auto_numbered = resolve_output_path(args.output, args.input)

    # For explicit .png paths, validate parent directory exists
    if not auto_numbered:
        output_dir = output_path.parent
        if str(output_dir) != "." and not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
            sys.exit(1)

    if not iris_bin.exists():
        print(f"Error: iris binary not found at {iris_bin}", file=sys.stderr)
        print("Run 'just' first to build iris.c", file=sys.stderr)
        sys.exit(1)

    if not model_dir.exists():
        model_name = "base model" if args.base else "model"
        print(f"Error: {model_name.capitalize()} not found at {model_dir}", file=sys.stderr)
        print("Run 'just' first to download the models", file=sys.stderr)
        sys.exit(1)

    # --- Dimension calculation (iteration 1 only) ---

    in_w, in_h, width, height = get_dimensions(args, args.input, args.max_area)

    # --- Evolution loop ---

    total = args.evolve
    evolving_input = args.input

    try:
        for i in range(1, total + 1):
            out_path = make_output_path(output_path, i, total)
            input_images = [evolving_input] + args.ref_images

            # Only pass dimensions on first iteration; subsequent ones let iris auto-detect
            iter_w = width if i == 1 else None
            iter_h = height if i == 1 else None

            if iter_w:
                dim_str = f" ({in_w}x{in_h} -> {iter_w}x{iter_h})"
            else:
                dim_str = f" ({in_w}x{in_h})"

            if total > 1:
                print(f"Iteration {i}/{total}: {evolving_input} -> {out_path}{dim_str}")
            else:
                print(f"Upscaling: {evolving_input} -> {out_path}{dim_str}")

            result = run_iris(
                iris_bin, model_dir, input_images, out_path, iter_w, iter_h, iris_passthrough
            )

            if result.returncode != 0:
                if total > 1:
                    print(
                        f"Error: iris failed on iteration {i}/{total} (exit code {result.returncode})",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Error: iris exited with code {result.returncode}",
                        file=sys.stderr,
                    )
                sys.exit(result.returncode)

            evolving_input = out_path

        if total > 1:
            print(f"Done: {total} iterations complete, final output: {evolving_input}")
        else:
            print(f"Done: {out_path}")

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
