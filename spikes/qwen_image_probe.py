#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.4.0",
#     "torchvision",
#     "transformers>=5.17",
#     "diffusers @ git+https://github.com/huggingface/diffusers",
#     "accelerate",
#     "pillow",
#     "click",
# ]
# ///
# ABOUTME: Throwaway spike that runs Qwen-Image-2.1 through diffusers on Apple Silicon (MPS)
# ABOUTME: Prints load time, per-step time, and peak memory for text-to-image or image-conditioned runs

import resource
import time
from pathlib import Path

import click
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-2.1"
GIB = 1024**3


def mps_gib():
    return torch.mps.driver_allocated_memory() / GIB


def peak_rss_gib():
    # ru_maxrss is bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB


@click.command()
@click.option("-i", "images", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="Condition image (repeatable). Omit for text-to-image.")
@click.option("-p", "--prompt", default="Create an exact copy of the input image.")
@click.option("-W", "--width", type=int, default=None)
@click.option("-H", "--height", type=int, default=None)
@click.option("--resolution", type=int, default=1024,
              help="Pipeline output_resolution: side length that condition images are resized to")
@click.option("-s", "--steps", type=int, default=40)
@click.option("-S", "--seed", type=int, default=42)
@click.option("--no-kv-cache", is_flag=True, help="Disable the prefix KV cache for text and condition images")
@click.option("-o", "output", type=click.Path(path_type=Path), default=Path("output/qwen-probe/out.png"))
def main(images, prompt, width, height, resolution, steps, seed, no_kv_cache, output):
    t0 = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("mps")
    torch.mps.synchronize()
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s  mps={mps_gib():.1f}GiB  peak_rss={peak_rss_gib():.1f}GiB")

    step_times = []
    last = [0.0]

    def on_step_end(pipe, step, timestep, callback_kwargs):
        torch.mps.synchronize()
        now = time.perf_counter()
        step_times.append(now - last[0])
        last[0] = now
        print(f"  step {step + 1}/{steps}: {step_times[-1]:.1f}s  mps={mps_gib():.1f}GiB")
        return callback_kwargs

    condition = [Image.open(p) for p in images]
    t1 = time.perf_counter()
    last[0] = t1
    result = pipe(
        prompt=prompt,
        image=condition or None,
        width=width,
        height=height,
        output_resolution=resolution,
        num_inference_steps=steps,
        use_kv_cache=not no_kv_cache,
        generator=torch.Generator("cpu").manual_seed(seed),
        callback_on_step_end=on_step_end,
    ).images[0]
    torch.mps.synchronize()
    done = time.perf_counter()
    total_s = done - t1
    decode_s = done - last[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)

    # The first step includes text/image encoding and the prefix prefill; the rest reuse the KV cache.
    steady = step_times[1:] or step_times
    print(f"generate: {total_s:.1f}s  first_step={step_times[0]:.1f}s  "
          f"steady_step={sum(steady) / len(steady):.1f}s  "
          f"vae_decode={decode_s:.1f}s")
    print(f"peak: mps={mps_gib():.1f}GiB  rss={peak_rss_gib():.1f}GiB")
    print(f"output: {output}  size={result.size}  mode={result.mode}")


if __name__ == "__main__":
    main()
