from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    index = json.loads(args.index.read_text())
    sidecar_root = Path(index["sidecar_root"]) / "episodes"
    stride = int(index["anchor_stride"])
    result = {
        "schema_version": 1,
        "index": str(args.index.resolve()),
        "splits": {},
        "test_locked_used": False,
    }
    for split in ("train", "viz_val"):
        visible_count = 0
        invisible_count = 0
        heights = []
        widths = []
        invisible_run_lengths = Counter()
        negative_global_indices = []
        small_bbox_global_indices = []
        descriptors = index["splits"][split]
        global_offset = 0
        for descriptor in descriptors:
            sidecar = json.loads(
                (sidecar_root / f'{descriptor["relative"]}.json').read_text()
            )
            labels = sidecar["steps"]
            run = 0
            anchors = range(
                int(descriptor["anchor_start"]),
                int(descriptor["anchor_end"]) + 1,
                stride,
            )
            for local_index, anchor in enumerate(anchors):
                global_index = global_offset + local_index
                label = labels[anchor]
                bbox = label.get("bbox_xyxy")
                visible = bool(label.get("visible") and bbox is not None)
                if visible:
                    visible_count += 1
                    if run:
                        invisible_run_lengths[run] += 1
                        run = 0
                    x1, y1, x2, y2 = (float(value) for value in bbox)
                    image_height = float(sidecar.get("image_height", 480))
                    image_width = float(sidecar.get("image_width", 640))
                    normalized_height = (
                        min(image_height, max(0.0, y2))
                        - min(image_height, max(0.0, y1))
                    ) / image_height
                    normalized_width = (
                        min(image_width, max(0.0, x2))
                        - min(image_width, max(0.0, x1))
                    ) / image_width
                    heights.append(normalized_height)
                    widths.append(normalized_width)
                    if normalized_height <= 0.65:
                        small_bbox_global_indices.append(global_index)
                else:
                    invisible_count += 1
                    negative_global_indices.append(global_index)
                    run += 1
            if run:
                invisible_run_lengths[run] += 1
            global_offset += int(descriptor["anchor_count"])
        total = visible_count + invisible_count
        height = np.asarray(heights, dtype=np.float64)
        width = np.asarray(widths, dtype=np.float64)
        result["splits"][split] = {
            "descriptors": len(descriptors),
            "anchors": total,
            "visible": visible_count,
            "invisible": invisible_count,
            "invisible_rate": invisible_count / max(1, total),
            "bbox_height_quantiles": {
                str(q): float(np.quantile(height, q))
                for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
            },
            "bbox_width_quantiles": {
                str(q): float(np.quantile(width, q))
                for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
            },
            "invisible_run_lengths": {
                str(length): count
                for length, count in sorted(invisible_run_lengths.items())
            },
            "negative_global_indices": negative_global_indices,
            "small_bbox_global_indices": small_bbox_global_indices,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
