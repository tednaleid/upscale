#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["click", "Pillow"]
# ///
# ABOUTME: Throwaway spike that upscales one image with NVIDIA PiD's from_clean mode (VAE encode, then pixel-diffusion decode)
# ABOUTME: Wraps the MPS-patched clone in .llm/PiD and its .venv-mps interpreter; mirrors upscale.py's -i/-o/-p/--scale/--count options

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
PID_DIR = REPO_DIR / ".llm" / "PiD"
PID_PYTHON = PID_DIR / ".venv-mps" / "bin" / "python"
PID_FACTOR = 4


def align16(n):
    return max(16, int(n) // 16 * 16)


def output_paths(output, input_stem, count):
    """An explicit .png path is used as-is for a single image; anything else is an auto-numbered base name."""
    if output and output.suffix == ".png" and count == 1:
        return [output]
    base = output.with_suffix("") if output else Path(f"{input_stem}-pid")
    paths = []
    n = 0
    while len(paths) < count:
        path = Path(f"{base}-{n:05d}.png")
        if not path.exists():
            paths.append(path)
        n += 1
    return paths


@click.command()
@click.option("-i", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="Input image")
@click.option("-o", "output", type=click.Path(path_type=Path), default=None,
              help="Output: a .png path (single image), or a base name that gets -NNNNN.png appended "
                   "(default base: <input stem>-pid in the current directory)")
@click.option("-p", "--prompt", required=True, help="Caption describing the image; PiD conditions on it")
@click.option("--scale", type=float, default=None,
              help="Output size as a percentage of the input (e.g. 200 = 2x). Default 400. "
                   "Mutually exclusive with --output-size.")
@click.option("--output-size", type=int, default=None,
              help="Long side of the output in pixels, aspect ratio preserved")
@click.option("--count", type=int, default=1, help="Generate N images with consecutive seeds (one model load)")
@click.option("--ckpt", type=click.Choice(["2k", "2kto4k_v1pt5"]), default="2k",
              help="PiD checkpoint: 2k for outputs near 2048, 2kto4k_v1pt5 for outputs up to 4096")
@click.option("-s", "--steps", type=int, default=4, help="PiD sampling steps (the checkpoints are 4-step distilled)")
@click.option("-S", "--seed", type=int, default=5, help="Seed of the first image")
def main(input_path, output, prompt, scale, output_size, count, ckpt, steps, seed):
    if scale and output_size:
        raise click.UsageError("--scale and --output-size are mutually exclusive")
    if not PID_PYTHON.exists():
        raise click.ClickException(f"PiD environment not found at {PID_PYTHON}")
    outputs = output_paths(output, input_path.stem, count)

    img = Image.open(input_path).convert("RGB")
    ratio = output_size / max(img.size) if output_size else (scale or 400) / 100
    # PiD always decodes at its trained factor, so the input is resized to the target size divided by that factor
    in_w = align16(img.width * ratio / PID_FACTOR)
    in_h = align16(img.height * ratio / PID_FACTOR)
    print(f"PiD {ckpt}: {input_path.name} {img.width}x{img.height} -> resized to {in_w}x{in_h} -> "
          f"{in_w * PID_FACTOR}x{in_h * PID_FACTOR} "
          f"({(in_w * PID_FACTOR // 16) * (in_h * PID_FACTOR // 16)} tokens) x{count}", file=sys.stderr)
    if not 384 <= max(in_w, in_h) <= 640:
        print(f"Warning: PiD input long side is {max(in_w, in_h)}px; the 2k checkpoint works best near 512px "
              f"(color casts near 256px, artifacts near 128px)", file=sys.stderr)
    if (in_w, in_h) != img.size:
        img = img.resize((in_w, in_h), Image.LANCZOS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prepared = tmp / "input.png"
        img.save(prepared)
        # One manifest line per image: PiD seeds each line with seed + line index
        manifest = tmp / "manifest.jsonl"
        manifest.write_text("".join(json.dumps({"image": str(prepared)}) + "\n" for _ in range(count)))
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        env.update(PYTHONPATH=".", PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
        t0 = time.monotonic()
        result = subprocess.run(
            [str(PID_PYTHON), "-m", "pid._src.inference.from_clean",
             "--backbone", "flux2", "--pid_ckpt_type", ckpt,
             "--manifest", str(manifest), "--prompt", prompt,
             "--degrade_sigmas", "0.0", "--cfg_scale", "1",
             "--pid_inference_steps", str(steps), "--scale", str(PID_FACTOR), "--seed", str(seed),
             "--save_format", "png", "--output_dir", str(tmp / "out")],
            cwd=PID_DIR, env=env, capture_output=True, text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stdout[-3000:] + result.stderr[-3000:])
            raise click.ClickException("PiD failed")
        # PiD also writes the cropped input and a plain VAE decode; its own results are under the flux2_* tag directory
        produced = sorted((tmp / "out").glob("flux2_*/sigma_*/*_input.png"))
        if len(produced) != count:
            raise click.ClickException(f"expected {count} PiD outputs, found: {produced}")
        for src, dst in zip(produced, outputs):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            print(f"Saved {dst}", file=sys.stderr)
    print(f"Done: {count} image(s) in {time.monotonic() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
