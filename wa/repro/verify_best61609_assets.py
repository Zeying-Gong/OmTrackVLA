"""Verify independently acquired WA61609 scene, humanoid and robot assets.

Read-only: no download, extraction, installation, or filesystem writes.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


OPTIONAL = {"source_mesh_not_proven_runtime_required", "use_unconfirmed"}
MANIFEST_SHA256 = "28a8310032b2ff0add984d86a2407da604552e781843bb459603bdb9f7c90ca5"
EXPECTED_COUNTS = {
    "hab_spot_arm": 25,
    "habitat_humanoids_100": 500,
    "hm3d_v0.2_config": 2,
    "hm3d_v0.2_val": 238,
    "humanoid_mapping": 1,
    "mp3d_habitat": 57,
    "supplemental_global_navmesh": 4,
}
EXPECTED_BYTES = {
    "hab_spot_arm": 3624467,
    "habitat_humanoids_100": 598735385,
    "hm3d_v0.2_config": 25413,
    "hm3d_v0.2_val": 5070254533,
    "humanoid_mapping": 32722,
    "mp3d_habitat": 3553043977,
    "supplemental_global_navmesh": 127996,
}
ROLES = {
    "hab_spot_arm": {"robot_asset"},
    "habitat_humanoids_100": {
        "render_config", "render_mesh", "skeleton", "motion",
        "source_mesh_not_proven_runtime_required",
    },
    "hm3d_v0.2_config": {"scene_dataset_config"},
    "hm3d_v0.2_val": {"primary", "scene_sidecar"},
    "humanoid_mapping": {"semantic_name_to_id"},
    "mp3d_habitat": {"primary", "scene_sidecar"},
    "supplemental_global_navmesh": {"use_unconfirmed"},
}
PREFIXES = {
    "hab_spot_arm": "data/robots/hab_spot_arm/",
    "habitat_humanoids_100": "data/humanoids/humanoid_data/",
    "hm3d_v0.2_config": "data/scene_datasets/hm3d/",
    "hm3d_v0.2_val": "data/scene_datasets/hm3d/val/",
    "mp3d_habitat": "data/scene_datasets/mp3d/",
    "supplemental_global_navmesh": "data/scene_datasets/navmeshes/",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        return False
    rel = PurePosixPath(name)
    return (not rel.is_absolute() and ".." not in rel.parts
            and "." not in name.split("/") and "" not in name.split("/")
            and str(rel) == name)


def within(path, directory):
    return path == directory or directory in path.parents


def validate_manifest(report):
    if not isinstance(report, dict):
        raise ValueError("asset manifest must be an object")
    if report.get("schema") != "wa61609_hash_only_asset_acquisition_v1":
        raise ValueError("unexpected asset manifest schema")
    if report.get("status") != "ORIGINAL_NAS_HASH_INVENTORY_NOT_PUBLIC_DOWNLOAD_VERIFIED":
        raise ValueError("asset manifest is not the completed SHA inventory")
    if report.get("episodes") != 4215:
        raise ValueError("fixed episode count is not 4215")
    if report.get("fixed_manifest_sha256") != "a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f":
        raise ValueError("fixed evaluation manifest identity changed")
    if report.get("counts_by_group") != EXPECTED_COUNTS or report.get("bytes_by_group") != EXPECTED_BYTES:
        raise ValueError("fixed asset group counts or bytes changed")
    rows = report.get("files")
    if not isinstance(rows, list) or len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError("fixed asset file count changed")
    counts = Counter()
    byte_counts = Counter()
    primary = []
    human_roles = defaultdict(Counter)
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("asset row must be an object")
        group, role, name = row.get("group"), row.get("role"), row.get("restore_path")
        if group not in ROLES or role not in ROLES[group]:
            raise ValueError(f"unexpected group/role: {group}/{role}")
        if not safe_relative(name) or name in names:
            raise ValueError(f"unsafe or duplicate restore path: {name}")
        names.add(name)
        if group == "humanoid_mapping":
            if name != "humanoid_infos.json":
                raise ValueError("humanoid index path changed")
        elif not name.startswith(PREFIXES[group]):
            raise ValueError(f"asset outside group prefix: {name}")
        if not isinstance(row.get("bytes"), int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0:
            raise ValueError(f"invalid size: {name}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid SHA256: {name}")
        counts[group] += 1
        byte_counts[group] += row["bytes"]
        if role == "primary":
            primary.append(name.removeprefix("data/scene_datasets/"))
        if group == "habitat_humanoids_100":
            parts = PurePosixPath(name).parts
            if len(parts) != 5:
                raise ValueError(f"unexpected humanoid leaf path: {name}")
            human_roles[parts[3]][role] += 1
    if dict(counts) != EXPECTED_COUNTS or dict(byte_counts) != EXPECTED_BYTES:
        raise ValueError("asset rows disagree with fixed group counts/bytes")

    scenes = report.get("scene_ids")
    if not isinstance(scenes, list) or len(scenes) != 101 or len(set(scenes)) != 101:
        raise ValueError("fixed 101 scene IDs missing or duplicated")
    for scene in scenes:
        if not safe_relative(scene):
            raise ValueError(f"unsafe scene ID: {scene}")
        parts = PurePosixPath(scene).parts
        if not ((len(parts) == 4 and parts[:2] == ("hm3d", "val") and parts[-1].endswith(".basis.glb"))
                or (len(parts) == 3 and parts[0] == "mp3d" and parts[-1].endswith(".glb"))):
            raise ValueError(f"unexpected scene prefix: {scene}")
    if Counter(scene.split("/", 1)[0] for scene in scenes) != {"hm3d": 87, "mp3d": 14}:
        raise ValueError("HM3D/MP3D scene count changed")
    if set(primary) != set(scenes) or len(primary) != len(scenes):
        raise ValueError("scene IDs do not match primary scene files")

    humans = report.get("humanoid_names")
    if not isinstance(humans, list) or len(humans) != 100 or len(set(humans)) != 100:
        raise ValueError("fixed 100 humanoid names missing or duplicated")
    for human in humans:
        if not isinstance(human, str) or re.fullmatch(r"[A-Za-z0-9_-]+", human) is None:
            raise ValueError(f"unsafe humanoid name: {human}")
    if set(humans) != set(human_roles):
        raise ValueError("humanoid names do not match asset leaves")
    if any(dict(roles) != {role: 1 for role in ROLES["habitat_humanoids_100"]}
           for roles in human_roles.values()):
        raise ValueError("each humanoid requires five distinct fixed roles")
    return rows


def contained_asset_path(path, project_root, allowed_scene_roots, is_scene):
    """Resolve before reading; external roots are explicit, scene-only allowlists."""
    resolved = path.resolve(strict=False)
    permitted = [project_root]
    if is_scene:
        permitted.extend(allowed_scene_roots)
    if not any(within(resolved, directory) for directory in permitted):
        raise ValueError(f"path escapes allowed roots: {path} -> {resolved}")
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True,
                    help="Root containing humanoid_infos.json and data/")
    ap.add_argument("--strict-all", action="store_true",
                    help="Also require source FBX and unconfirmed global navmeshes")
    ap.add_argument("--strict-extras", action="store_true",
                    help="Fail on extra files in fixed scene/humanoid/robot leaf directories")
    ap.add_argument("--allow-scene-root", action="append", type=Path, default=[],
                    help="Explicit external scene dataset root; repeat for nested symlink roots")
    args = ap.parse_args()
    manifest_digest = sha256(args.manifest)
    if manifest_digest != MANIFEST_SHA256:
        raise ValueError(f"manifest SHA256 mismatch: expected {MANIFEST_SHA256}, got {manifest_digest}")
    report = json.loads(args.manifest.read_text())
    rows = validate_manifest(report)
    root = args.project_root.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed_scene_roots = []
    for candidate in args.allow_scene_root:
        directory = candidate.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError(f"scene root is not a directory: {candidate}")
        allowed_scene_roots.append(directory)

    issues = []
    optional_missing = []
    known = set()
    verified = 0
    for row in rows:
        name = row["restore_path"]
        known.add(name)
        expected_hash = row["sha256"]
        path = root / name
        is_scene = name.startswith("data/scene_datasets/")
        try:
            contained_asset_path(path, root, allowed_scene_roots, is_scene)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append({"status": "UNAPPROVED_PATH", "path": name, "error": str(exc)})
            continue
        required = args.strict_all or row["role"] not in OPTIONAL
        if not path.is_file():
            (issues if required else optional_missing).append({"status": "MISSING" if required else "OPTIONAL_MISSING", "path": name})
            continue
        try:
            actual_size = path.stat().st_size
            if actual_size != row["bytes"]:
                issues.append({"status": "SIZE_MISMATCH", "path": name, "expected_bytes": row["bytes"], "actual_bytes": actual_size})
                continue
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                issues.append({"status": "HASH_MISMATCH", "path": name, "expected_sha256": expected_hash, "actual_sha256": actual_hash})
                continue
        except OSError as exc:
            issues.append({"status": "READ_ERROR", "path": name, "error": str(exc)})
            continue
        verified += 1

    leaf_dirs = set()
    for sid in report["scene_ids"]:
        leaf_dirs.add(str(PurePosixPath("data/scene_datasets") / PurePosixPath(sid).parent))
    for name in report["humanoid_names"]:
        leaf_dirs.add("data/humanoids/humanoid_data/" + name)
    leaf_dirs.add("data/robots/hab_spot_arm")
    extras = []
    for leaf in sorted(leaf_dirs):
        directory = root / leaf
        try:
            contained_asset_path(directory, root, allowed_scene_roots,
                                 leaf.startswith("data/scene_datasets/"))
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append({"status": "UNAPPROVED_LEAF", "path": leaf, "error": str(exc)})
            continue
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    try:
                        contained_asset_path(path, root, allowed_scene_roots,
                                             rel.startswith("data/scene_datasets/"))
                    except (OSError, RuntimeError, ValueError) as exc:
                        issues.append({"status": "UNAPPROVED_EXTRA_PATH", "path": rel, "error": str(exc)})
                        continue
                    if rel not in known:
                        extras.append(rel)
    if args.strict_extras:
        issues.extend({"status": "EXTRA", "path": rel} for rel in extras)

    result = {
        "status": "PASS" if not issues else "FAIL",
        "manifest_sha256": manifest_digest,
        "allowed_scene_roots": [str(path) for path in allowed_scene_roots],
        "project_root": str(root),
        "expected_files": len(report["files"]),
        "verified_files": verified,
        "required_issues": issues,
        "optional_missing": optional_missing,
        "extra_files": extras,
        "extra_files_are_fatal": args.strict_extras,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
