#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/data/kodak24/clean"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$TARGET_DIR"

echo "Downloading Kodak24 dataset..."
if command -v curl >/dev/null 2>&1; then
  curl -L "http://r0k.us/graphics/kodak/kodak.zip" -o "$TMP_DIR/kodak.zip"
elif command -v wget >/dev/null 2>&1; then
  wget "http://r0k.us/graphics/kodak/kodak.zip" -O "$TMP_DIR/kodak.zip"
else
  echo "Error: please install curl or wget." >&2
  exit 1
fi

unzip -q "$TMP_DIR/kodak.zip" -d "$TMP_DIR/kodak"
find "$TMP_DIR/kodak" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -exec cp {} "$TARGET_DIR" \;

echo "Kodak24 images copied to: $TARGET_DIR"
