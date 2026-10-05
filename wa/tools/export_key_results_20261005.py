#!/usr/bin/env python3
"""Export small, hash-verified result snapshots; never alter evaluation outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
OUT = Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61259/task_72336/wa_initial_bbox_repair_v2")
EXPECTED = {
    "combined_episodes.jsonl": "517937840d3e5da9b42717cf3c16dadbce499bf1c6c203173badb6c8f3b9c08e",
    "summary.json": "3c6ef1a2039e557aa564a7bdbca7ce0e3a616246f5c483991b137c640138b3c2",
}
for name, sha in EXPECTED.items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == sha, name
rows = [json.loads(line) for line in (OUT / "combined_episodes.jsonl").read_text().splitlines()]
summary = json.loads((OUT / "summary.json").read_text())
assert len(rows) == len({(r["task"], r["key"]) for r in rows}) == 4215
counts = {}
for task in ("stt", "dt", "at"):
    group = [r for r in rows if r["task"] == task]
    counts[task] = {
        "episodes": len(group),
        "success": sum(int(r["success"]) for r in group),
        "human_collision": sum(int(r["collision"]) for r in group),
        "invalid_init": sum(not r["policy_init_valid"] for r in group),
    }
    assert len(group) == 1405
    for metric, key in (("SR", "success"), ("CR", "human_collision")):
        assert abs(100 * counts[task][key] / 1405 - summary["metrics_percent"][task][metric]) < 1e-10
assert sum(x["success"] for x in counts.values()) == 3627
assert sum(x["invalid_init"] for x in counts.values()) == 0
audit_path = ROOT / "artifacts/initial_bbox_repair_review_job_61259/audit.json"
audit = json.loads(audit_path.read_text())
assert audit["status"] == "PASS" and audit["reused_unchanged"] == 4208 and audit["new"] == 7
repo = Path(__file__).resolve().parents[2]
result = {
    "updated_date": "2026-10-05",
    "source_job_task": "61259/72336",
    "source_output": str(OUT),
    "source_sha256": EXPECTED,
    "counts": counts,
    "total_success": 3627,
    "total_episodes": 4215,
    "overall_sr_percent": 100 * 3627 / 4215,
    "summary": summary,
    "audit": audit,
    "interpretation": [
        "Existing validation, not untouched test; no full LightNav comparison.",
        "Seven actual replacement outcomes plus 4208 unchanged baseline rows.",
        "First GT bbox repair only; subsequent semantic visibility scoring unchanged.",
        "Score changes across annotation protocols are not training gains.",
        "TR retains actual-step fallback for 52 missing references per task; macro_TR is separate.",
    ],
}
dest = repo / "wa/results/KEY_RESULTS_20261005.json"
dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"status": "PASS", "counts": counts, "output": str(dest)}, ensure_ascii=False))
