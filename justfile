# ABOUTME: Build and setup tasks for the Flux.2 Klein super-resolution upscaler
# ABOUTME: Manages iris.c cloning, building, model downloads, and installation

iris_dir := justfile_directory() / "iris.c"
bin_link := home_directory() / ".local" / "bin" / "upscale"

# Clone iris.c, build, and download all models (9B requires HF_TOKEN, skipped without it)
default: iris-build download-4b download-9b

# Clone iris.c or pull/rebase latest main
iris-pull:
    #!/usr/bin/env bash
    set -eu
    if [ -d "{{ iris_dir }}" ]; then
        echo "iris.c already cloned, pulling latest..."
        cd "{{ iris_dir }}"
        git pull --rebase
    else
        echo "Cloning iris.c..."
        git clone https://github.com/antirez/iris.c.git "{{ iris_dir }}"
    fi

# Build iris.c for the current platform
iris-build: iris-pull
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
