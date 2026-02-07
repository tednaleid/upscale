#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow"]
# ///
# ABOUTME: Wrapper script for flux2.c super-resolution upscaling with smart scaling and evolution
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

    Returns (in_w, in_h, out_w, out_h). out_w/out_h are None when flux should auto-detect.
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


def make_output_path(base_output, iteration, total):
    """Return the output path for this iteration.

    Single iteration: use base_output directly.
    Multiple iterations: use {stem}_{NNN}.png numbering.
    """
    base_output = base_output.with_suffix(".png")
    if total == 1:
        return base_output
    return base_output.with_stem(f"{base_output.stem}_{iteration:03d}")


def run_flux(flux_bin, model_dir, input_images, output_path, width, height, extra_args):
    """Build and run a single flux command. Returns the CompletedProcess."""
    cmd = [str(flux_bin), "-d", str(model_dir)]

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
    flux_bin = script_dir / "flux2.c" / "flux"
    model_dir = script_dir / "flux-klein-model"

    parser = argparse.ArgumentParser(
        description="Upscale an image using Flux.2 Klein 4B super-resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect dimensions, default prompt copies input
  %(prog)s input.png output.png

  # Scale to 200%% (2x), dimensions aligned to 16px
  %(prog)s input.png output.png --scale 200

  # Explicit width, height inferred from aspect ratio
  %(prog)s input.png output.png -W 1024

  # Custom prompt for guided upscaling
  %(prog)s input.png output.png -p "A sharp, detailed photograph"

  # Evolve 3 iterations with a persistent reference image
  %(prog)s input.png output.png -i ref.png --evolve 3 -p "blend with reference"

  # Show in terminal, verbose, 8 steps
  %(prog)s input.png output.png --show -v -s 8
""",
    )

    parser.add_argument("input", type=Path, help="Path to input image")
    parser.add_argument("output", type=Path, help="Path for output image (always .png)")
    parser.add_argument(
        "-i",
        dest="ref_images",
        type=Path,
        action="append",
        default=[],
        help="Additional reference image (repeatable, passed to every iteration)",
    )
    parser.add_argument(
        "-W", "--width", type=int, default=None, help="Output width (default: flux auto-detect)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=None, help="Output height (default: flux auto-detect)"
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

    flux_group = parser.add_argument_group(
        "Common flux options (passed through to flux)"
    )
    flux_group.add_argument(
        "-p", "--prompt", default="Create an exact copy of the input image.",
        help='Text prompt (default: "Create an exact copy of the input image.")',
    )
    flux_group.add_argument(
        "-s", "--steps", type=int, default=None,
        help="Sampling steps (default: 4, more = better quality)",
    )
    flux_group.add_argument(
        "-S", "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    flux_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed output",
    )
    flux_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Silent mode, no output",
    )
    flux_group.add_argument(
        "--show", action="store_true",
        help="Display image in terminal (Kitty/Ghostty/iTerm2)",
    )
    flux_group.add_argument(
        "--show-steps", action="store_true",
        help="Display each denoising step (slower)",
    )

    args, extra_args = parser.parse_known_args()

    # Rebuild known flux flags into the passthrough list
    flux_passthrough = ["-p", args.prompt]
    if args.steps is not None:
        flux_passthrough.extend(["-s", str(args.steps)])
    if args.seed is not None:
        flux_passthrough.extend(["-S", str(args.seed)])
    if args.verbose:
        flux_passthrough.append("-v")
    if args.quiet:
        flux_passthrough.append("-q")
    if args.show:
        flux_passthrough.append("--show")
    if args.show_steps:
        flux_passthrough.append("--show-steps")
    flux_passthrough.extend(extra_args)

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

    output_dir = args.output.parent
    if str(output_dir) != "." and not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    if not flux_bin.exists():
        print(f"Error: flux binary not found at {flux_bin}", file=sys.stderr)
        print("Run ./setup.sh first to build flux2.c", file=sys.stderr)
        sys.exit(1)

    if not model_dir.exists():
        print(f"Error: Model not found at {model_dir}", file=sys.stderr)
        print("Run ./setup.sh first to download the model", file=sys.stderr)
        sys.exit(1)

    # --- Dimension calculation (iteration 1 only) ---

    in_w, in_h, width, height = get_dimensions(args, args.input, args.max_area)

    # --- Evolution loop ---

    total = args.evolve
    evolving_input = args.input

    try:
        for i in range(1, total + 1):
            out_path = make_output_path(args.output, i, total)
            input_images = [evolving_input] + args.ref_images

            # Only pass dimensions on first iteration; subsequent ones let flux auto-detect
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

            result = run_flux(
                flux_bin, model_dir, input_images, out_path, iter_w, iter_h, flux_passthrough
            )

            if result.returncode != 0:
                if total > 1:
                    print(
                        f"Error: flux failed on iteration {i}/{total} (exit code {result.returncode})",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Error: flux exited with code {result.returncode}",
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
