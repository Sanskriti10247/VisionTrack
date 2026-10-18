#!/usr/bin/env bash
# Downloads the public datasets used for evaluation into ./data.
#   MOT17 (train split has tracking ground truth)  ~5.5 GB  CC BY-NC-SA 3.0
#   COCO val2017 images + annotations              ~1.0 GB  CC BY 4.0 (annotations)
# Usage: scripts/download_data.sh [mot17] [coco]   (default: both)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(mot17 coco)

fetch() {  # url dest
  if [ -s "$2" ]; then echo "have $2"; else curl -L --fail -C - -o "$2.part" "$1" && mv "$2.part" "$2"; fi
}

for t in "${targets[@]}"; do
  case "$t" in
    mot17)
      fetch https://motchallenge.net/data/MOT17.zip data/raw/MOT17.zip
      # Images are identical across the DPM/FRCNN/SDP copies; the FRCNN copy is enough.
      unzip -q -n data/raw/MOT17.zip 'MOT17/train/*-FRCNN/*' -d data
      python scripts/prepare_mot17.py
      ;;
    coco)
      fetch http://images.cocodataset.org/zips/val2017.zip data/raw/val2017.zip
      fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip data/raw/annotations_trainval2017.zip
      mkdir -p data/coco
      unzip -q -n data/raw/val2017.zip -d data/coco
      unzip -q -n data/raw/annotations_trainval2017.zip 'annotations/instances_val2017.json' -d data/coco
      ;;
    *) echo "unknown target $t" >&2; exit 2 ;;
  esac
done
echo "done"
