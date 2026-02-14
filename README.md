# Flux.2 Klein Super Resolution Upscaler

Use the Flux.2 Klein 4B model as a super-resolution upscaler, based on [antirez's technique](https://bsky.app/profile/antirez.bsky.social/post/3mdyj7tbmoc2c).

## How It Works

Flux.2 Klein uses "in-context conditioning" - instead of adding noise to the input image like traditional img2img, it passes the reference image as tokens that the transformer attends to during generation. This allows the model to intelligently upscale while preserving composition and details.

The key insight is using a prompt like "Create an exact copy of the input image" with a larger output resolution than the input, effectively turning a generative model into a super-resolution tool.

## Requirements

- **macOS** (Apple Silicon recommended) or **Linux**
- ~32GB disk space for 4B models (~16GB each), ~60GB for 9B models (~30GB each)
- ~4-5GB RAM minimum (with memory mapping)
- For Linux with BLAS: `sudo apt install libopenblas-dev`

## Setup

Requires [just](https://github.com/casey/just). Run once to clone iris.c, build it, and download models:

```bash
# Downloads all models — 9B included if HF_TOKEN is set, skipped otherwise
just

# To include 9B models, set HF_TOKEN first
export HF_TOKEN=hf_...
just

# Individual model downloads
just download-4b              # both 4B models (~32GB)
just download-4b-distilled    # 4B distilled only (~16GB)
just download-4b-base         # 4B base only (~16GB)
just download-9b              # both 9B models (~60GB, needs HF_TOKEN)
just download-9b-distilled    # 9B distilled only (~30GB, needs HF_TOKEN)
just download-9b-base         # 9B base only (~30GB, needs HF_TOKEN)

# Add upscale to your PATH
just install              # symlinks to ~/.local/bin/upscale
just uninstall            # removes the symlink
```

This will:
1. Clone [antirez/iris.c](https://github.com/antirez/iris.c)
2. Build with the optimal backend for your platform (MPS for Apple Silicon, BLAS for others)
3. Download the selected models

The 9B models require a [HuggingFace token](https://huggingface.co/settings/tokens) and accepting the [FLUX.2-klein-9B license](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).

## Usage

### Basic Upscaling

```bash
# Auto-number output: creates input-00000.png (or next free number)
./upscale.py input.png

# Scale to 200% (2x), dimensions aligned to 16px boundaries
./upscale.py input.png --scale 200

# Explicit width, height inferred from input aspect ratio
./upscale.py input.png -W 1024

# Explicit .png path: always overwrites on subsequent runs
./upscale.py input.png output.png
```

### Output Naming

The output argument is optional and controls naming behavior:

| Command | Output file |
|---------|------------|
| `./upscale.py input.png` | `input-00000.png` (auto-increments) |
| `./upscale.py input.png results/` | `results/input-00000.png` (dir created if needed) |
| `./upscale.py input.png upscaled` | `upscaled-00000.png` (auto-increments) |
| `./upscale.py input.png output.png` | `output.png` (overwrites) |

Auto-numbered outputs never overwrite — the number increments to find the next free filename. Only explicit `.png` paths overwrite.

### Base Model

The base (undistilled) model produces higher quality results at the cost of speed (~25x slower). Iris auto-detects steps (50) and guidance (4.0) from the model.

```bash
# Base model - higher quality, slower
./upscale.py input.png --base

# Base with linear schedule and fewer steps for a quick preview
./upscale.py input.png --base --linear -s 10

# Base with custom guidance
./upscale.py input.png --base -g 6.0
```

### 9B Model

The 9B model is larger and may produce better results. Combine with `--base` for the highest quality variant:

```bash
# 9B distilled
./upscale.py input.png --9b

# 9B base (highest quality, slowest)
./upscale.py input.png --9b --base
```

### Options

```
Positional:
  input                 Path to input image
  output                Output path (optional, .png = overwrite, no ext = auto-number, / = directory)

Upscale options:
  --base                Use base model (higher quality, ~25x slower)
  --9b                  Use 9B model (larger, non-commercial). Download with: HF_TOKEN=... just download-9b
  -W, --width N         Output width (default: iris auto-detect from input)
  -H, --height N        Output height (default: iris auto-detect from input)
  --scale N             Scale percentage (e.g. 200 = 2x). Mutually exclusive with -W/-H.
  -i PATH               Additional reference image (repeatable, passed to every iteration)
  --evolve N            Number of evolution iterations (default: 1)
  --count N             Generate N images with different seeds, model loaded once (default: 1)
  --max-area N          Pixel area warning threshold (default: 1048576 = 1024x1024)

Generation options:
  -p, --prompt TEXT     Text prompt (default: "Create an exact copy of the input image.")
  -s, --steps N         Sampling steps (default: auto, 4 distilled / 50 base)
  -g, --guidance N      CFG guidance scale (default: auto, 1.0 distilled / 4.0 base)
  -S, --seed N          Random seed for reproducibility
  --linear              Use linear timestep schedule (faster preview with fewer steps)
  --power               Use power curve timestep schedule
  --power-alpha N       Power schedule exponent (default: 2.0)
  --show                Display final image in terminal (Kitty/Ghostty/iTerm2/WezTerm)
  --show-steps          Display each denoising step in terminal (slower)
```

### Dimension Handling

When `-W`/`-H` are omitted, iris auto-detects dimensions from the input image. The wrapper adds smart options on top:

- **`--scale 200`** — reads input dimensions, multiplies by 2x, aligns to 16px boundaries
- **`-W 1024`** (width only) — infers height from the input's aspect ratio, aligned to 16px
- **`-H 768`** (height only) — infers width from the input's aspect ratio, aligned to 16px
- **Neither** — iris auto-detects from input; the wrapper still checks area for warnings

A warning is printed to stderr if the output area exceeds `--max-area` (default 1MP = 1024x1024).

### Evolution

Use `--evolve N` to iteratively refine an image by feeding each output back as the next input:

```bash
# 3 evolution iterations — produces input-00000_001.png, input-00000_002.png, input-00000_003.png
./upscale.py input.png --evolve 3
```

Each iteration feeds the previous output as the primary input to iris. Dimensions are only set on the first iteration; subsequent iterations let iris auto-detect from the previous output.

### Multi-Seed Generation

Use `--count N` to generate N images with different random seeds. The model is loaded once and reused for all images, saving ~35s per additional image:

```bash
# Generate 3 variations
./upscale.py input.png --count 3

# First image uses seed 42, remaining 2 get random seeds
./upscale.py input.png --count 3 --seed 42

# Combine with evolution: 2 seeds x 3 evolve = 6 images total
./upscale.py input.png --count 2 --evolve 3
```

Each seed iteration gets its own auto-numbered output base (e.g. `input-00000.png`, `input-00001.png`, `input-00002.png`). When combined with `--evolve`, each seed's evolution iterations are numbered within its base.

### Persistent Reference Images

Use `-i` to pass additional reference images that persist across all evolution iterations:

```bash
# Blend with a style reference across 3 iterations
./upscale.py input.png -i style.png --evolve 3 -p "blend with reference style"
```

This runs:
```
Iteration 1: iris -i input.png              -i style.png -o input-00000_001.png -p "..."
Iteration 2: iris -i input-00000_001.png    -i style.png -o input-00000_002.png -p "..."
Iteration 3: iris -i input-00000_002.png    -i style.png -o input-00000_003.png -p "..."
```

Multiple `-i` flags can be used to pass several reference images.

### Examples

```bash
# Simple upscale, auto-named output
./upscale.py small.png

# 4x upscale with higher quality, custom base name
./upscale.py thumb.png large --scale 400 -s 8

# Width-only, aspect ratio preserved, explicit overwrite path
./upscale.py photo.png wide.png -W 2048 -p "high resolution landscape"

# Evolve with style reference, output to directory
./upscale.py sketch.png results/ -i style.png --evolve 5 --show -p "oil painting style"
```

## Direct iris.c Usage

For more control, use the iris binary directly:

```bash
./iris.c/iris -d flux-klein-4b \
    -i input.png \
    -W 1024 -H 1024 \
    -o output.png \
    -p "Create an exact copy of the input image."
```

### Additional iris.c Features

```bash
# Multi-reference editing (combine images)
./iris.c/iris -d flux-klein-4b \
    -i car.png -i beach.png \
    -p "sports car on beach" \
    -o result.png

# Interactive mode
./iris.c/iris -d flux-klein-4b

# Display in terminal (Kitty/iTerm2/Ghostty)
./iris.c/iris -d flux-klein-4b -p "A cat" --show
```

## Tips

- **Prompt matters**: While "Create an exact copy" works well, describing the desired output ("A high resolution photograph with sharp details") can sometimes produce better results.
- **More steps = better quality**: Distilled default is 4 steps (fast). Try 8-12 for higher quality at the cost of speed.
- **Base model**: Use `--base` for the highest quality. Combine with `--linear -s 10` for a faster preview.
- **9B model**: Use `--9b` for the larger model. Combine with `--base` for the absolute best quality (slowest).
- **Aspect ratio**: When using `-W` or `-H` alone, the other dimension is inferred from the input aspect ratio and aligned to 16px boundaries.
- **Evolution**: Multiple iterations can progressively refine details. Start with 2-3 and increase if needed.
- **Memory**: Uses memory-mapped weights by default, keeping RAM usage low.

## Credits

- [antirez/iris.c](https://github.com/antirez/iris.c) - Pure C inference engine for Flux.2
- [Black Forest Labs](https://blackforestlabs.ai/) - Flux.2 Klein model
- [Original technique](https://bsky.app/profile/antirez.bsky.social/post/3mdyj7tbmoc2c) by antirez
