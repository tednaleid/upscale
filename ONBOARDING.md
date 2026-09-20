# Onboarding

`upscale` is a Python CLI that turns the Flux.2 Klein image model into a super-resolution upscaler. It prompts the model to "create an exact copy of the input image" at a larger output size, using in-context image conditioning. Inference runs in a patched fork of antirez's pure-C engine `iris.c`, loaded as a shared library through ctypes. It also does text-to-image, multi-reference generation, iterative evolution, and multi-seed batches.

## Stack
- Language: Python 3.12+ as `uv` single-file scripts (inline dependencies: click, Pillow); no pyproject.toml
- Inference engine: `iris.c` (C, Metal/MPS on Apple Silicon, BLAS elsewhere), cloned from the `tednaleid/iris.c` fork. The `iris-dylib` recipe hardcodes `.mps.o` objects and Metal frameworks, so the shared library that `upscale.py` needs only links on Apple Silicon.
- Models: Flux.2 Klein 4B and 9B, each in distilled and base variants, in `flux-klein-*` directories at the repo root
- Task runner: `just` (authoritative)
- CI/CD: none
- Lint, typecheck, format: not configured

## Common commands
- Full setup (clone and build iris.c, link dylib, download models): `just`
- Update and rebuild iris.c: `just iris-update`
- Check `iris.h` against the ctypes bindings: `just iris-check`
- Relink the shared library only: `just iris-dylib`
- Download one model: `just download-4b-distilled` (see `just --list` for the other variants; 9B needs `HF_TOKEN`)
- Test: `just test` (smoke test, runs real generations with the 4B distilled model, writes to `output/test-<timestamp>/`)
- Install on PATH: `just install` (symlinks `upscale.py` to `~/.local/bin/upscale`)

## Architecture
`upscale.py` is a click group. Invoked without a subcommand it generates images; `info` prints the metadata embedded in a generated PNG and `show` displays PNGs in the terminal. Generation resolves dimensions and output paths, then drives `IrisContext` from `iris_ffi.py`, which wraps `iris.c/libiris.dylib` with ctypes. With `--count` greater than 1, text embeddings and image latents are encoded once and reused across seeds through two functions that exist only in the fork (`iris_img2img_precomputed`, `iris_multiref_precomputed`). `check_iris_api.py` diffs `iris.c/iris.h` against `iris_api.snapshot` so upstream API changes are caught before the ctypes bindings break at runtime.

The model directory and the dylib are both located relative to the real path of `upscale.py`, so the symlinked install works from any directory.

## Key paths
- `upscale.py` -- CLI entry point: option parsing, dimension and output-path logic, generation loops, PNG metadata injection
- `iris_ffi.py` -- ctypes structs and function signatures, `IrisContext`, progress callbacks, terminal image display
- `check_iris_api.py` and `iris_api.snapshot` -- API drift check for `iris.h`
- `justfile` -- setup, build, download, test, and install recipes
- `iris.c/` -- gitignored clone of the fork; build output `libiris.dylib` lives here
- `flux-klein-4b/`, `flux-klein-4b-base/`, `flux-klein-9b/`, `flux-klein-9b-base/` -- gitignored model weights
- `input.png`, `avatar.png` -- fixtures used by `just test`; gitignored by the `*.png` rule, so they are not in a fresh clone
- `output/` -- gitignored generation and test output
- `.llm/` -- gitignored reference clones for research

## How to run
```
just
./upscale.py -i input.png --scale 200 --show
```

## Dig deeper
- [README.md](./README.md) -- full CLI usage, option reference, output naming rules, tips
- [IRIS_PATCHES.md](./IRIS_PATCHES.md) -- the fork's patches, why they exist, and the rebase workflow against upstream
