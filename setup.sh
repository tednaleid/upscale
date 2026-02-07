#!/bin/bash
# ABOUTME: Setup script for flux2.c - clones repo, builds binary, and downloads model
# ABOUTME: Run once to set up the upscaling environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUX_DIR="$SCRIPT_DIR/flux2.c"
MODEL_DIR="$SCRIPT_DIR/flux-klein-model"

echo "=== flux2.c Super Resolution Setup ==="
echo ""

# Clone flux2.c if not present
if [ -d "$FLUX_DIR" ]; then
    echo "flux2.c already cloned, pulling latest..."
    cd "$FLUX_DIR"
    git pull
else
    echo "Cloning flux2.c..."
    git clone https://github.com/antirez/flux2.c.git "$FLUX_DIR"
    cd "$FLUX_DIR"
fi

# Detect platform and build
echo ""
echo "Building flux2.c..."
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "Detected Apple Silicon - building with Metal (MPS)..."
        make mps
    else
        echo "Detected Intel Mac - building with BLAS..."
        make blas
    fi
else
    # Linux
    if ldconfig -p 2>/dev/null | grep -q openblas || [ -f /usr/lib/libopenblas.so ]; then
        echo "Detected OpenBLAS - building with BLAS..."
        make blas
    else
        echo "No BLAS found - building generic (consider: sudo apt install libopenblas-dev)..."
        make generic
    fi
fi

echo ""
echo "Build complete!"

# Download model if not present
if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "Model directory already exists at: $MODEL_DIR"
    echo "Skipping download. Delete the directory and re-run to re-download."
else
    echo ""
    echo "Downloading model (~16GB)..."
    echo "This will take a while depending on your connection speed."
    echo ""

    # Use the download script from flux2.c
    cd "$FLUX_DIR"

    # Check if huggingface_hub is available for Python download
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using Python downloader..."
        python3 download_model.py
    elif command -v curl &>/dev/null; then
        echo "Using shell downloader..."
        ./download_model.sh
    else
        echo "ERROR: Need either 'huggingface_hub' Python package or 'curl' to download model"
        echo "Install with: pip install huggingface_hub"
        echo "Or install curl"
        exit 1
    fi

    # Move model to our directory
    if [ -d "$FLUX_DIR/flux-klein-model" ]; then
        mv "$FLUX_DIR/flux-klein-model" "$MODEL_DIR"
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Model location: $MODEL_DIR"
echo "Binary location: $FLUX_DIR/flux"
echo ""
echo "Test with:"
echo "  ./upscale.sh input.png output.png"
echo ""
echo "Or directly:"
echo "  $FLUX_DIR/flux -d $MODEL_DIR -i input.png -W 1024 -H 1024 -o output.png -p 'Create an exact copy of the input image.'"
