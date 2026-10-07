"""Freeze all 126 best-61609 STT failures for bounded same-state recovery.

No simulation, model inference, source edits or training happen here. A plan is
evaluation-set adaptation, not an untouched test or a demonstration release.
"""
import argparse
import hashlib
import json
from pathlib import Path

from wa.wm.failure_state_protocol import build_plan, load_plan


def write_exclusive(path, plan):
    """Create only; never replace an existing plan (including dangling symlinks)."""
    data = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--student-root", required=True)
    p.add_argument("--bbox-plan", required=True)
    p.add_argument("--output")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if a.check_only == bool(a.output):
        p.error("choose exactly one of --check-only or --output")
    if a.output and (Path(a.output).exists() or Path(a.output).is_symlink()):
        raise FileExistsError("refusing to overwrite: "+a.output)
    plan = build_plan(a.manifest, a.student_root, a.bbox_plan)
    if a.check_only:
        print(json.dumps(dict(status="SOURCE_AND_PLAN_CHECK_PASS_NOT_WRITTEN",
            experiment=plan["experiment"], entries=len(plan["entries"]),
            lanes=[len(x) for x in plan["lanes"]], protocol_sha256=plan["protocol_sha256"],
            source_files=len(plan["source_hashes"]), training_released=False)))
        return
    digest = write_exclusive(a.output, plan)
    # Re-read exact output and original sources after publication. An exception
    # retains the file for diagnosis; this command never overwrites or deletes.
    load_plan(a.output, digest)
    print(json.dumps(dict(status="FROZEN_PLAN_ONLY", path=str(Path(a.output).resolve()),
        sha256=digest, entries=len(plan["entries"]), lanes=[len(x) for x in plan["lanes"]],
        protocol_sha256=plan["protocol_sha256"], training_released=False)))


if __name__ == "__main__":
    main()
