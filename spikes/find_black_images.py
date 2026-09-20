#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
# ABOUTME: Throwaway spike that scans a directory tree for PNGs that are entirely or nearly black
# ABOUTME: Prints each PNG's size, max pixel value, and embedded upscale/iris metadata

import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
for path in sorted(root.rglob("*.png")):
    img = Image.open(path)
    max_value = max(hi for _, hi in img.convert("RGB").getextrema())
    verdict = "BLACK" if max_value <= 8 else "ok"
    meta = {k: v for k, v in img.info.items() if k.startswith(("upscale:", "iris:"))}
    print(f"{verdict}\t{path}\t{img.size}\tmax={max_value}")
    if verdict == "BLACK":
        for key, value in meta.items():
            print(f"\t{key} = {value}")
