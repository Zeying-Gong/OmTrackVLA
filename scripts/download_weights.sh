#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/nas_ray/home/zeying.gong/venvs/omtrackvla-modular/bin/python}"

DETECTOR_PATH="$ROOT_DIR/models/torchvision/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
DETECTOR_SHA256="907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5"
REID_PATH="$ROOT_DIR/models/reid/osnet_x0_25_msmt17.pt"
REID_SHA256="6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"

has_checksum() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] && [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

download_detector() {
  if has_checksum "$DETECTOR_PATH" "$DETECTOR_SHA256"; then
    echo "[weights] detector already verified"
    return
  fi
  mkdir -p "$(dirname "$DETECTOR_PATH")"
  curl -L --fail --retry 3 \
    -o "$DETECTOR_PATH.part" \
    "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
  has_checksum "$DETECTOR_PATH.part" "$DETECTOR_SHA256"
  mv "$DETECTOR_PATH.part" "$DETECTOR_PATH"
}

download_reid() {
  if has_checksum "$REID_PATH" "$REID_SHA256"; then
    echo "[weights] OSNet ReID already verified"
    return
  fi
  mkdir -p "$(dirname "$REID_PATH")"
  if ! timeout 90 "$PYTHON_BIN" -m gdown \
    1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF -O "$REID_PATH.part"; then
    echo "[weights] Google Drive unavailable; using a checksum-verified mirror"
    curl -L --fail --retry 3 -o "$REID_PATH.part" \
      "https://ghfast.top/https://raw.githubusercontent.com/TaskerJang/lost-and-find/master/osnet_x0_25_msmt17.pt"
  fi
  has_checksum "$REID_PATH.part" "$REID_SHA256"
  mv "$REID_PATH.part" "$REID_PATH"
}

download_detector
download_reid
sha256sum "$DETECTOR_PATH" "$REID_PATH"
