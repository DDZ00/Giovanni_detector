#!/usr/bin/env bash
# Download Giovanni's GGUF models into ./giovanni/models/.
#
# These URLs follow the Qwen team's official Hugging Face GGUF release pattern.
# Verify them against https://huggingface.co/Qwen before running, or override
# via env vars: GIOVANNI_BASE_GGUF_URL / GIOVANNI_INSTRUCT_GGUF_URL.
#
# Disk: ~1.0 GB each (Q4_K_M quant). Run once per host; the read-only volume
# mount means re-running just overwrites the files in place.
set -euo pipefail

BASE_URL="${GIOVANNI_BASE_GGUF_URL:-https://huggingface.co/QuantFactory/Qwen2.5-1.5B-GGUF/resolve/main/Qwen2.5-1.5B.Q4_K_M.gguf}"
INSTRUCT_URL="${GIOVANNI_INSTRUCT_GGUF_URL:-https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
LLAMA_URL="${GIOVANNI_LLAMA_GGUF_URL:-https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"

# Falcon-7B is the canonical scoring/sampling model from the FastDetectGPT
# paper (Bao et al. 2023). Same-family pair so tokenizers align exactly.
# Disk: ~4.5 GB each. Set GIOVANNI_FETCH_FALCON=1 to download.
FALCON_BASE_URL="${GIOVANNI_FALCON_BASE_GGUF_URL:-https://huggingface.co/maddes8cht/tiiuae-falcon-7b-gguf/resolve/main/tiiuae-falcon-7b-Q4_K_M.gguf}"
FALCON_INSTRUCT_URL="${GIOVANNI_FALCON_INSTRUCT_GGUF_URL:-https://huggingface.co/maddes8cht/tiiuae-falcon-7b-instruct-gguf/resolve/main/tiiuae-falcon-7b-instruct-Q4_K_M.gguf}"

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$DEST_DIR"

fetch() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "✔ already present: $dest"
        return
    fi
    echo "↓ fetching $url"
    curl -L --fail --progress-bar -o "$dest.partial" "$url"
    mv "$dest.partial" "$dest"
    echo "✔ saved: $dest"
}

# Same-family pair (legacy "qwen" preset)
fetch "$BASE_URL"     "$DEST_DIR/qwen2.5-1.5b-q4_k_m.gguf"
fetch "$INSTRUCT_URL" "$DEST_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Cross-family partner (new "cross" preset: Qwen base + Llama instruct)
fetch "$LLAMA_URL"    "$DEST_DIR/llama-3.2-3b-instruct-q4_k_m.gguf"

# FastDetectGPT-original pair (opt-in: ~9 GB combined)
if [[ "${GIOVANNI_FETCH_FALCON:-0}" == "1" ]]; then
    fetch "$FALCON_BASE_URL"     "$DEST_DIR/falcon-7b-q4_k_m.gguf"
    fetch "$FALCON_INSTRUCT_URL" "$DEST_DIR/falcon-7b-instruct-q4_k_m.gguf"
fi

echo
echo "Done. GGUFs present:"
ls -lh "$DEST_DIR"/*.gguf
sha256sum "$DEST_DIR"/*.gguf
