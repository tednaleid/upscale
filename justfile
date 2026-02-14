# ABOUTME: Build and setup tasks for the Flux.2 Klein super-resolution upscaler
# ABOUTME: Manages iris.c cloning, building, model downloads, and installation

iris_dir := justfile_directory() / "iris.c"
bin_link := home_directory() / ".local" / "bin" / "upscale"

# Clone/update iris.c, build, and download all models (9B requires HF_TOKEN, skipped without it)
default: iris-update download-4b download-9b

# Pull latest iris.c, verify API compatibility, rebuild, and link dylib
iris-update: iris-pull iris-check iris-build iris-dylib

# Clone iris.c or pull/rebase latest
iris-pull:
    #!/usr/bin/env bash
    set -eu
    if [ -d "{{ iris_dir }}" ]; then
        echo "iris.c already cloned, pulling latest..."
        cd "{{ iris_dir }}"
        git pull --rebase
    else
        echo "Cloning iris.c..."
        git clone git@github.com:tednaleid/iris.c.git "{{ iris_dir }}"
    fi

# Build iris.c for the current platform (assumes iris-pull already ran)
iris-build:
    #!/usr/bin/env bash
    set -eu
    cd "{{ iris_dir }}"
    echo "Building iris.c..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            echo "Detected Apple Silicon - building with Metal (MPS)..."
            make mps
        else
            echo "Detected Intel Mac - building with BLAS..."
            make blas
        fi
    else
        if ldconfig -p 2>/dev/null | grep -q openblas || [ -f /usr/lib/libopenblas.so ]; then
            echo "Detected OpenBLAS - building with BLAS..."
            make blas
        else
            echo "No BLAS found - building generic (consider: sudo apt install libopenblas-dev)..."
            make generic
        fi
    fi
    echo "Build complete!"

# Check if iris.h API surface has changed from what iris_ffi.py expects
iris-check:
    #!/usr/bin/env bash
    set -eu
    python3 "{{ justfile_directory() }}/check_iris_api.py" || {
        echo ""
        echo "  iris_ffi.py may need updating to match the new API."
        echo "  After verifying, run: python3 check_iris_api.py --update"
        echo ""
        exit 1
    }

# Link shared library for Python ctypes usage (assumes iris-build already ran)
# NOTE: Apple Silicon/MPS specific — hardcodes .mps.o suffixes and Metal frameworks
iris-dylib:
    #!/usr/bin/env bash
    set -eu
    cd "{{ iris_dir }}"
    echo "Building libiris.dylib..."
    gcc -shared -o libiris.dylib \
        iris.mps.o iris_kernels.mps.o iris_tokenizer.mps.o iris_vae.mps.o \
        iris_transformer_flux.mps.o iris_transformer_zimage.mps.o \
        iris_sample.mps.o iris_image.mps.o jpeg.mps.o iris_safetensors.mps.o \
        iris_qwen3.mps.o iris_qwen3_tokenizer.mps.o terminals.mps.o \
        iris_metal.o \
        -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
        -framework MetalPerformanceShadersGraph -framework Foundation -lm
    echo "Built: {{ iris_dir }}/libiris.dylib"

# Download both 4B models (distilled + base, ~32GB total)
download-4b: download-4b-distilled download-4b-base

# Download both 9B models (distilled + base, ~60GB total, requires HF_TOKEN)
download-9b: download-9b-distilled download-9b-base

# Download 4B distilled model (~16GB)
download-4b-distilled: iris-pull (_download "4b")

# Download 4B base model (~16GB)
download-4b-base: iris-pull (_download "4b-base")

# Download 9B distilled model (~30GB, requires HF_TOKEN)
download-9b-distilled: iris-pull (_download "9b")

# Download 9B base model (~30GB, requires HF_TOKEN)
download-9b-base: iris-pull (_download "9b-base")

# Symlink upscale.py to ~/.local/bin/upscale
install:
    #!/usr/bin/env bash
    set -eu
    mkdir -p "$(dirname "{{ bin_link }}")"
    ln -sf "{{ justfile_directory() }}/upscale.py" "{{ bin_link }}"
    echo "Installed: {{ bin_link }} -> upscale.py"

# Remove the ~/.local/bin/upscale symlink
uninstall:
    #!/usr/bin/env bash
    set -eu
    if [ -L "{{ bin_link }}" ]; then
        rm "{{ bin_link }}"
        echo "Removed: {{ bin_link }}"
    else
        echo "No symlink found at {{ bin_link }}"
    fi

# Smoke test: exercises all generation code paths and CLI switches
test:
    #!/usr/bin/env bash
    set -eu
    d="output/test-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$d"
    echo "Test output: $d"

    run() { echo ""; echo "=== $1 ==="; shift; "$@"; }

    # Single-ref, auto-detect dimensions, auto-numbered output
    run "basic upscale" ./upscale.py -i input.png -o "$d/basic"

    # Scale + explicit .png + seed + linear schedule
    run "scale+linear+seed" ./upscale.py -i input.png --scale 110 --linear -S 42 -o "$d/scale.png"

    # Width-only + power schedule + custom prompt
    run "width+power+prompt" ./upscale.py -i input.png -W 288 --power --power-alpha 3.0 -p "sharp" -o "$d/width"

    # Height-only + -o flag + custom steps
    run "height+oflag+steps" ./upscale.py -i input.png -H 288 -s 2 -o "$d/height.png"

    # Both W+H + custom guidance
    run "both-dims+guidance" ./upscale.py -i input.png -W 288 -H 288 -g 2.0 -o "$d/both"

    # Evolution (2 iterations, tests output chaining)
    run "evolve" ./upscale.py -i input.png -s 2 --evolve 2 -o "$d/evolve"

    # Count (2 seeds, triggers precomputed single-ref path)
    run "count" ./upscale.py -i input.png -s 2 --count 2 -o "$d/count"

    # Count + evolve combined (precomputed + evolve, first_img_cache reuse)
    run "count+evolve" ./upscale.py -i input.png -s 2 --count 2 --evolve 2 -o "$d/countevolve"

    # Multi-reference, non-precomputed (iris_multiref with 2 refs)
    run "multiref" ./upscale.py -i input.png -i avatar.png -p "blend" -W 288 -H 288 -o "$d/multiref"

    # Multi-reference + count (precomputed multi-ref with persistent_ref_latents)
    run "multiref+count" ./upscale.py -i input.png -i avatar.png -p "blend" -W 288 -H 288 -s 2 --count 2 -o "$d/multiref-count"

    # Text-to-image (iris_generate, no input images)
    run "txt2img" ./upscale.py -p "a small red circle" -W 256 -H 256 -o "$d/txt2img"

    # Directory output with trailing /
    run "dir-output" ./upscale.py -i input.png -o "$d/dir/"

    echo ""
    echo "=== All tests passed ==="
    echo "Output: $d"
    ls -la "$d"/ "$d/dir/" 2>/dev/null || true

# Remove iris.c build artifacts
iris-clean:
    #!/usr/bin/env bash
    set -eu
    if [ -d "{{ iris_dir }}" ]; then
        cd "{{ iris_dir }}"
        make clean 2>/dev/null || true
        echo "Cleaned iris.c build artifacts"
    else
        echo "iris.c not cloned, nothing to clean"
    fi

[private]
_download variant:
    #!/usr/bin/env bash
    set -eu
    model_dir="{{ justfile_directory() }}/flux-klein-{{ variant }}"
    if [ -d "$model_dir" ] && [ "$(ls -A "$model_dir" 2>/dev/null)" ]; then
        echo "Model already exists at: $model_dir"
        echo "Delete the directory and re-run to re-download."
        exit 0
    fi
    # 9B models require a HuggingFace token — skip gracefully without one
    token_args=""
    case "{{ variant }}" in
        9b|9b-base)
            if [ -z "${HF_TOKEN:-}" ]; then
                echo "Skipping {{ variant }} model (no HF_TOKEN set)."
                echo "  export HF_TOKEN=hf_... && just download-9b"
                echo "  Create a token at https://huggingface.co/settings/tokens"
                echo "  Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
                exit 0
            fi
            token_args="--token $HF_TOKEN"
            ;;
    esac
    echo "Downloading {{ variant }} model..."
    cd "{{ iris_dir }}"
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using Python downloader..."
        python3 download_model.py {{ variant }} $token_args
    elif command -v curl &>/dev/null; then
        echo "Using shell downloader..."
        ./download_model.sh {{ variant }} $token_args
    else
        echo "ERROR: Need either 'huggingface_hub' Python package or 'curl' to download model"
        echo "Install with: pip install huggingface_hub"
        exit 1
    fi
    # Move model from iris.c subdirectory to project root
    if [ -d "{{ iris_dir }}/flux-klein-{{ variant }}" ]; then
        mv "{{ iris_dir }}/flux-klein-{{ variant }}" "$model_dir"
    fi
    echo "Downloaded to: $model_dir"
