#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/data/kodak24/clean"
KODAK_BASE_URL="https://r0k.us/graphics/kodak/kodak"

mkdir -p "$TARGET_DIR"

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 15 "$url" -o "$out"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=15 "$url" -O "$out"
    return
  fi

  echo "Error: please install curl or wget." >&2
  exit 1
}

echo "Downloading Kodak24 dataset (24 PNG files)..."
failed=0
for idx in $(seq -w 1 24); do
  fname="kodim${idx}.png"
  url="${KODAK_BASE_URL}/${fname}"
  out="${TARGET_DIR}/${fname}"

  echo "  - ${fname}"
  if ! download_file "$url" "$out"; then
    echo "Failed to download: $url" >&2
    failed=1
  fi
done

img_count="$(find "$TARGET_DIR" -maxdepth 1 -type f -name 'kodim*.png' | wc -l | tr -d ' ')"
if [[ "$img_count" -ne 24 || "$failed" -ne 0 ]]; then
  echo "Error: expected 24 files, got ${img_count}. Please re-run and check network access." >&2
  exit 1
fi

echo "Kodak24 downloaded successfully to: $TARGET_DIR"
