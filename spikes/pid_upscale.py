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
@click.option("--scale", type=int, default=400,
              help="Scale percentage, a multiple of 100 (default 400; the checkpoints are trained for 4x)")
@click.option("--output-size", type=int, default=None,
              help="Long side of the output in pixels; the input is resized to output-size / scale first")
@click.option("--input-size", type=int, default=None,
              help="Resize the input so its long side is N pixels before upscaling (the 2k checkpoint expects about 512)")
@click.option("--count", type=int, default=1, help="Generate N images with consecutive seeds (one model load)")
@click.option("--ckpt", type=click.Choice(["2k", "2kto4k_v1pt5"]), default="2k",
              help="PiD checkpoint: 2k for outputs near 2048, 2kto4k_v1pt5 for outputs up to 4096")
@click.option("-s", "--steps", type=int, default=4, help="PiD sampling steps (the checkpoints are 4-step distilled)")
@click.option("-S", "--seed", type=int, default=5, help="Seed of the first image")
def main(input_path, output, prompt, scale, output_size, input_size, count, ckpt, steps, seed):
    if scale % 100 != 0 or scale < 100:
        raise click.UsageError("--scale must be a multiple of 100 (PiD takes an integer factor)")
    if output_size and input_size:
        raise click.UsageError("--output-size and --input-size are mutually exclusive")
    if not PID_PYTHON.exists():
        raise click.ClickException(f"PiD environment not found at {PID_PYTHON}")
    factor = scale // 100
    if output_size:
        input_size = output_size // factor
    outputs = output_paths(output, input_path.stem, count)

    img = Image.open(input_path).convert("RGB")
    if input_size:
        ratio = input_size / max(img.size)
        img = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS)
    # PiD center-crops each side down to a multiple of 16
    in_w, in_h = (img.width // 16) * 16, (img.height // 16) * 16
    print(f"PiD {ckpt}: {input_path.name} {in_w}x{in_h} -> {in_w * factor}x{in_h * factor} "
          f"({(in_w * factor // 16) * (in_h * factor // 16)} tokens) x{count}", file=sys.stderr)

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
             "--pid_inference_steps", str(steps), "--scale", str(factor), "--seed", str(seed),
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
