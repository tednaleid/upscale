#!/bin/bash
# ABOUTME: Setup script for iris.c - clones repo, builds binary, and downloads models
# ABOUTME: Run once to set up the upscaling environment (distilled + base models)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRIS_DIR="$SCRIPT_DIR/iris.c"

# Parse arguments
MODEL_SIZE="4b"
if [[ "$1" == "--9b" ]]; then
    MODEL_SIZE="9b"
fi

MODEL_DIR="$SCRIPT_DIR/flux-klein-$MODEL_SIZE"
BASE_MODEL_DIR="$SCRIPT_DIR/flux-klein-${MODEL_SIZE}-base"

echo "=== iris.c Super Resolution Setup ==="
echo ""

# Check HF_TOKEN for 9B models
TOKEN_ARGS=""
if [[ "$MODEL_SIZE" == "9b" ]]; then
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: 9B models require a HuggingFace token."
        echo "Set HF_TOKEN in your environment:"
        echo "  export HF_TOKEN=hf_..."
        echo ""
        echo "You can create a token at https://huggingface.co/settings/tokens"
        echo "The 9B models also require accepting the license at:"
        echo "  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
        exit 1
    fi
    TOKEN_ARGS="--token $HF_TOKEN"
fi

# Clone iris.c if not present
if [ -d "$IRIS_DIR" ]; then
    echo "iris.c already cloned, pulling latest..."
    cd "$IRIS_DIR"
    git pull
else
    echo "Cloning iris.c..."
    git clone https://github.com/antirez/iris.c.git "$IRIS_DIR"
    cd "$IRIS_DIR"
fi

# Detect platform and build
echo ""
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
    echo "Downloading ${MODEL_SIZE} model..."
    echo "This will take a while depending on your connection speed."
    echo ""

    # Use the download script from iris.c
    cd "$IRIS_DIR"

    # Check if huggingface_hub is available for Python download
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using Python downloader..."
        python3 download_model.py $MODEL_SIZE $TOKEN_ARGS
    elif command -v curl &>/dev/null; then
        echo "Using shell downloader..."
        ./download_model.sh $MODEL_SIZE $TOKEN_ARGS
    else
        echo "ERROR: Need either 'huggingface_hub' Python package or 'curl' to download model"
        echo "Install with: pip install huggingface_hub"
        echo "Or install curl"
        exit 1
    fi

    # Move model to our directory
    if [ -d "$IRIS_DIR/flux-klein-$MODEL_SIZE" ]; then
        mv "$IRIS_DIR/flux-klein-$MODEL_SIZE" "$MODEL_DIR"
    fi
fi

# Download base model if not present
if [ -d "$BASE_MODEL_DIR" ] && [ "$(ls -A "$BASE_MODEL_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "Base model directory already exists at: $BASE_MODEL_DIR"
    echo "Skipping download. Delete the directory and re-run to re-download."
else
    echo ""
    echo "Downloading ${MODEL_SIZE}-base model..."
    echo "This will take a while depending on your connection speed."
    echo ""

    cd "$IRIS_DIR"

    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using Python downloader..."
        python3 download_model.py ${MODEL_SIZE}-base $TOKEN_ARGS
    elif command -v curl &>/dev/null; then
        echo "Using shell downloader..."
        ./download_model.sh ${MODEL_SIZE}-base $TOKEN_ARGS
    else
        echo "ERROR: Need either 'huggingface_hub' Python package or 'curl' to download model"
        echo "Install with: pip install huggingface_hub"
        echo "Or install curl"
        exit 1
    fi

    # Move base model to our directory
    if [ -d "$IRIS_DIR/flux-klein-${MODEL_SIZE}-base" ]; then
        mv "$IRIS_DIR/flux-klein-${MODEL_SIZE}-base" "$BASE_MODEL_DIR"
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Distilled model: $MODEL_DIR"
echo "Base model:      $BASE_MODEL_DIR"
echo "Binary location: $IRIS_DIR/iris"
echo ""
echo "Test with:"
echo "  ./upscale.py input.png output.png          # distilled (fast, ~5s)"
echo "  ./upscale.py input.png output.png --base    # base (higher quality, slower)"
