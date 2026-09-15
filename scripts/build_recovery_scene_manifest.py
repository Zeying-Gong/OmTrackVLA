#!/usr/bin/env python3
"""Freeze and verify scene-isolated recovery manifests for a development pilot.

No data discovery is performed. The first build receives explicit sample paths
and writes a frozen scene plan; subsequent builds must reuse that plan. The
independent candidate validator is imported only when validation is requested.
This does not grant formal-training or product-acceptance eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SCOPE = "development_pilot"
PLAN_STAGE = "recovery_scene_plan_v1"
MANIFEST_STAGE = "recovery_scene_manifest_v1"
DEFAULT_TASKS = ("STT", "DT", "AT")
Validator = Callable[[Path], Mapping[str, Any]]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_locked(value: str | Path) -> None:
    if "test_locked" in str(value).replace("\\", "/").casefold():
        raise ValueError("test_locked artifacts are forbidden")


def _path(value: str | Path, *, base: Path | None = None) -> Path:
    _reject_locked(value)
    result = Path(value).expanduser()
    if not result.is_absolute() and base is not None:
        result = base / result
    result = result.resolve(strict=True)
    _reject_locked(result)
    if not result.is_file():
        raise ValueError(f"expected an explicit file: {result}")
    return result


def artifact(value: str | Path) -> dict[str, str]:
    path = _path(value)
    return {"path": str(path), "sha256": sha256(path)}


def _check_artifact(value: Mapping[str, Any]) -> Path:
    path = _path(value["path"])
    if sha256(path) != value.get("sha256"):
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: hashlib.sha256(_json_bytes(value)).hexdigest()}


def _read_sealed(path: str | Path, *, stage: str, field: str) -> dict[str, Any]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sealed document must be a JSON object")
    unsigned = {key: item for key, item in value.items() if key != field}
    if hashlib.sha256(_json_bytes(unsigned)).hexdigest() != value.get(field):
        raise ValueError(f"{stage} checksum mismatch")
    if (value.get("schema_version") != 1 or value.get("stage") != stage
            or value.get("scope") != SCOPE or value.get("test_locked_used") is not False
            or value.get("formal_training_eligible") is not False):
        raise ValueError(f"invalid {stage} scope or eligibility")
    return value


def _write_new(path: str | Path, value: Mapping[str, Any]) -> Path:
    result = Path(path).expanduser().resolve()
    _reject_locked(result)
    result.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is deliberate: validation must never be reselected by
    # accidentally rerunning a build over the original split.
    with result.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + "\n")
    return result


def canonical_scene_id(value: Any) -> str:
    """Map common scene aliases to the physical scene, ignoring task/path roots."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a nonempty scene_id is required")
    _reject_locked(value)
    text = value.strip().replace("\\", "/").casefold()
    matches = {int(item) for item in re.findall(r"(?<![a-z0-9])scene[_-]?(\d+)(?!\d)", text)}
    if len(matches) > 1:
        raise ValueError(f"ambiguous scene aliases in scene_id: {value}")
    if matches:
        return f"scene{next(iter(matches)):05d}"
    name = text.rstrip("/").split("/")[-1]
    for suffix in (".scene_instance.json", ".basis.glb", ".glb", ".gltf", ".ply", ".json"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    if not name or name in {".", ".."}:
        raise ValueError("scene_id does not identify a scene")
    # HM3D folder IDs carry an ordinal that the mesh basename omits, e.g.
    # 00083-16tymPtM7uS/16tymPtM7uS.basis.glb. Both identify the same scene.
    hm3d_directory = re.fullmatch(r"\d{5}-([a-z0-9]{11})", name)
    if hm3d_directory:
        return hm3d_directory.group(1)
    return name


def _identity(evidence: Mapping[str, Any]) -> dict[str, Any]:
    index = evidence.get("dataset_index")
    anchor = evidence.get("anchor_environment_step")
    if (type(index) is not int or index < 0 or type(anchor) is not int or anchor < 1):
        raise ValueError("dataset_index and positive anchor must be integers")
    task = evidence.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task is required")
    return {"task": task.strip().upper(), "scene_id": canonical_scene_id(evidence["scene_id"]),
            "dataset_index": index, "anchor_environment_step": anchor}


def _key(identity: Mapping[str, Any]) -> str:
    return _json_bytes(identity).decode("utf-8")


def default_validator(sample_path: Path, *, artifact_root: Path | None = None) -> Mapping[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from omtrackvla.data.recovery_sequence import validate_recovery_sample
    return validate_recovery_sample(sample_path, artifact_root=artifact_root)


def _validate_samples(paths: Iterable[str | Path], validator: Validator) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identities: set[str] = set()
    seen_paths: set[Path] = set()
    for value in paths:
        path = _path(value)
        if path in seen_paths:
            raise ValueError(f"duplicate sample path: {path}")
        seen_paths.add(path)
        evidence = dict(validator(path))
        checks = evidence.get("checks")
        if (evidence.get("schema_version") != 2 or evidence.get("source_split") != "train"
                or evidence.get("formal_training_eligible") is not False
                or evidence.get("test_locked_used") is not False or not isinstance(checks, dict)
                or not checks or any(passed is not True for passed in checks.values())):
            raise ValueError("candidate lacks independent schema-v2 train admission evidence")
        identity = _identity(evidence)
        identity_key = _key(identity)
        if identity_key in identities:
            raise ValueError(f"duplicate physical sample identity: {identity_key}")
        identities.add(identity_key)
        artifacts = evidence.get("artifacts", {})
        for name in ("sample", "report", "rollout"):
            if name not in artifacts:
                raise ValueError(f"validator evidence lacks {name} hash")
            resolved = _check_artifact(artifacts[name])
            if name == "sample" and resolved != path:
                raise ValueError("validator evidence refers to a different sample")
        for media in artifacts.get("media", []):
            _check_artifact(media)
        result.append({"identity": identity, "legacy_sample_id": evidence.get("sample_id"),
                       "source_scene_id": evidence["scene_id"], "source_split": "train",
                       "artifacts": artifacts, "validation_evidence": evidence})
    if not result:
        raise ValueError("explicit sample list must not be empty")
    return sorted(result, key=lambda record: _key(record["identity"]))


def _coverage(records: list[dict[str, Any]], roles: Mapping[str, str],
              required_tasks: Iterable[str], min_train_scenes: int,
              min_val_scenes: int) -> dict[str, Any]:
    if type(min_train_scenes) is not int or type(min_val_scenes) is not int:
        raise ValueError("minimum scene counts must be integers")
    if min_train_scenes < 1 or min_val_scenes < 1:
        raise ValueError("both partitions must require at least one scene")
    required = sorted({str(task).strip().upper() for task in required_tasks})
    if not required or "" in required:
        raise ValueError("a nonempty required task set is mandatory")
    actual_scenes = {record["identity"]["scene_id"] for record in records}
    if set(roles) != actual_scenes or any(role not in {"train", "val"} for role in roles.values()):
        raise ValueError("scene plan must assign every actual scene exactly once to train or val")
    summary: dict[str, Any] = {"required_tasks": required, "task_requirement_scope": "global"}
    tasks = {record["identity"]["task"] for record in records}
    missing = sorted(set(required) - tasks)
    summary["global"] = {"sample_count": len(records), "scene_count": len(actual_scenes),
                         "tasks": sorted(tasks), "missing_required_tasks": missing}
    if missing:
        raise ValueError(f"required task coverage is incomplete: {missing}; observed {sorted(tasks)}")
    for role, minimum in (("train", min_train_scenes), ("val", min_val_scenes)):
        scenes = sorted(scene for scene, assigned in roles.items() if assigned == role)
        selected = [record for record in records if record["identity"]["scene_id"] in scenes]
        selected_tasks = sorted({record["identity"]["task"] for record in selected})
        summary[role] = {"scene_count": len(scenes), "sample_count": len(selected),
                         "scenes": scenes, "tasks": selected_tasks,
                         "missing_required_tasks": sorted(set(required) - set(selected_tasks)),
                         "minimum_scene_count": minimum}
        if len(scenes) < minimum or not selected:
            raise ValueError(f"{role} has {len(scenes)} scenes, requires at least {minimum}")
    return summary


def _prior_records(paths: Iterable[str | Path]) -> tuple[list[dict[str, str]], dict[str, str]]:
    bound = []
    old_roles: dict[str, str] = {}
    for path in paths:
        value = _read_sealed(path, stage=MANIFEST_STAGE, field="manifest_sha256")
        bound.append(artifact(path))
        for record in value["samples"]:
            scene = canonical_scene_id(record["identity"]["scene_id"])
            role = record["partition_role"]
            if role not in {"train", "val"} or (scene in old_roles and old_roles[scene] != role):
                raise ValueError("prior manifests disagree on permanent scene partition")
            old_roles[scene] = role
    return bound, old_roles


def freeze_scene_plan(*, sample_paths: Iterable[str | Path], output: str | Path,
                      parent_checkpoint: str | Path, collection_protocol: str | Path,
                      validator: Validator = default_validator, seed: int = 42,
                      val_fraction: float = 0.34, min_train_scenes: int = 2,
                      min_val_scenes: int = 2, required_tasks: Iterable[str] = DEFAULT_TASKS,
                      prior_manifests: Iterable[str | Path] = ()) -> dict[str, Any]:
    if Path(output).expanduser().exists():
        raise FileExistsError(f"refusing to overwrite frozen scene plan: {output}")
    if type(seed) is not int or not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("integer seed and val_fraction strictly between zero and one required")
    records = _validate_samples(sample_paths, validator)
    parent = artifact(parent_checkpoint)
    protocol = artifact(collection_protocol)
    prior, old_roles = _prior_records(prior_manifests)
    scenes = sorted({record["identity"]["scene_id"] for record in records})
    if len(scenes) < min_train_scenes + min_val_scenes:
        raise ValueError(f"need at least {min_train_scenes + min_val_scenes} distinct physical scenes")
    # Hash ranking is deterministic across Python versions and input ordering.
    ranked = sorted(scenes, key=lambda scene: (hashlib.sha256(f"{seed}\0{scene}".encode()).hexdigest(), scene))
    roles = {scene: old_roles[scene] for scene in scenes if scene in old_roles}
    target_val = max(min_val_scenes, min(len(scenes) - min_train_scenes,
                                      math.ceil(len(scenes) * val_fraction)))
    val_count = sum(role == "val" for role in roles.values())
    for scene in ranked:
        if scene not in roles:
            roles[scene] = "val" if val_count < target_val else "train"
            val_count += roles[scene] == "val"
    coverage = _coverage(records, roles, required_tasks, min_train_scenes, min_val_scenes)
    value = _seal({"schema_version": 1, "stage": PLAN_STAGE, "scope": SCOPE,
                   "status": "frozen", "formal_training_eligible": False, "test_locked_used": False,
                   "seed": seed, "partition_algorithm": "sha256_seed_and_canonical_scene_v1",
                   "val_fraction": val_fraction, "parent_checkpoint": parent,
                   "collection_protocol": protocol, "prior_manifests": prior,
                   "scene_partitions": dict(sorted(roles.items())), "coverage": coverage,
                   "validation_policy": "permanent_development_holdout_never_optimizer_input",
                   "samples": [{key: record[key] for key in ("identity", "artifacts")} for record in records]},
                  "plan_sha256")
    _write_new(output, value)
    return value


def build_manifest(*, plan_path: str | Path, output: str | Path,
                   validator: Validator = default_validator) -> dict[str, Any]:
    if Path(output).expanduser().exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    plan = _read_sealed(plan_path, stage=PLAN_STAGE, field="plan_sha256")
    if plan.get("status") != "frozen":
        raise ValueError("scene plan must be frozen")
    _check_artifact(plan["parent_checkpoint"])
    _check_artifact(plan["collection_protocol"])
    for prior in plan["prior_manifests"]:
        _check_artifact(prior)
    records = _validate_samples([record["artifacts"]["sample"]["path"] for record in plan["samples"]], validator)
    if [{key: record[key] for key in ("identity", "artifacts")} for record in records] != plan["samples"]:
        raise ValueError("validated identities or artifact hashes changed since the scene plan was frozen")
    roles = plan["scene_partitions"]
    for record in records:
        record["partition_role"] = roles[record["identity"]["scene_id"]]
        record["optimizer_input_allowed"] = record["partition_role"] == "train"
        record["admission_scope"] = SCOPE
    expected = plan["coverage"]
    coverage = _coverage(records, roles, expected["required_tasks"],
                         expected["train"]["minimum_scene_count"], expected["val"]["minimum_scene_count"])
    if coverage != expected:
        raise ValueError("frozen scene plan coverage mismatch")
    _, old_roles = _prior_records([prior["path"] for prior in plan["prior_manifests"]])
    if any(scene in roles and roles[scene] != role for scene, role in old_roles.items()):
        raise ValueError("a previously frozen scene cannot move between train and validation")
    value = _seal({"schema_version": 1, "stage": MANIFEST_STAGE, "scope": SCOPE,
                   "status": "admitted_for_development_pilot", "formal_training_eligible": False,
                   "test_locked_used": False, "plan": artifact(plan_path),
                   "parent_checkpoint": plan["parent_checkpoint"],
                   "collection_protocol": plan["collection_protocol"], "coverage": coverage,
                   "source_dataset_split": "train", "validation_policy": plan["validation_policy"],
                   "product_acceptance_evidence": False, "sample_count": len(records), "samples": records},
                  "manifest_sha256")
    _write_new(output, value)
    return value


def verify_manifest(path: str | Path, *, validator: Validator = default_validator,
                    role: str | None = None, for_training: bool = False) -> tuple[dict[str, Any], list[Path]]:
    """Revalidate immutable artifacts; callers must explicitly request their role."""
    if role not in {None, "train", "val"} or (for_training and role != "train"):
        raise ValueError("optimizer input requires the explicit train role")
    value = _read_sealed(path, stage=MANIFEST_STAGE, field="manifest_sha256")
    if value.get("status") != "admitted_for_development_pilot" or value.get("product_acceptance_evidence") is not False:
        raise ValueError("manifest is not admitted for a development pilot")
    plan_path = _check_artifact(value["plan"])
    plan = _read_sealed(plan_path, stage=PLAN_STAGE, field="plan_sha256")
    if plan.get("status") != "frozen":
        raise ValueError("scene plan must be frozen")
    for field in ("parent_checkpoint", "collection_protocol"):
        if value[field] != plan[field]:
            raise ValueError(f"manifest {field} does not match frozen plan")
        _check_artifact(value[field])
    for prior in plan["prior_manifests"]:
        _check_artifact(prior)
    if value.get("source_dataset_split") != "train" or len(value["samples"]) != value.get("sample_count"):
        raise ValueError("manifest source split or sample count mismatch")
    checked = _validate_samples([record["artifacts"]["sample"]["path"] for record in value["samples"]], validator)
    records = sorted(value["samples"], key=lambda record: _key(record["identity"]))
    if [{key: record[key] for key in ("identity", "artifacts")} for record in checked] != plan["samples"]:
        raise ValueError("manifest samples differ from frozen plan")
    roles: dict[str, str] = {}
    for observed, record in zip(checked, records):
        scene = observed["identity"]["scene_id"]
        assigned = record.get("partition_role")
        if (assigned not in {"train", "val"} or roles.setdefault(scene, assigned) != assigned
                or plan["scene_partitions"].get(scene) != assigned):
            raise ValueError("scene leakage or changed frozen scene partition")
        if (record.get("optimizer_input_allowed") is not (assigned == "train")
                or record.get("admission_scope") != SCOPE
                or any(record.get(key) != observed[key] for key in observed)):
            raise ValueError("manifest record or independent validation evidence changed")
    expected = plan["coverage"]
    coverage = _coverage(records, roles, expected["required_tasks"], expected["train"]["minimum_scene_count"],
                         expected["val"]["minimum_scene_count"])
    if coverage != value["coverage"] or coverage != expected:
        raise ValueError("manifest coverage differs from frozen plan")
    _, old_roles = _prior_records([prior["path"] for prior in plan["prior_manifests"]])
    if any(scene in roles and roles[scene] != assigned for scene, assigned in old_roles.items()):
        raise ValueError("a previously validated scene cannot become training data")
    selected = [_path(record["artifacts"]["sample"]["path"]) for record in records
                if role is None or record["partition_role"] == role]
    return value, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="freeze a plan using explicit sample files only")
    freeze.add_argument("--sample", type=Path, action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--parent-checkpoint", type=Path, required=True)
    freeze.add_argument("--collection-protocol", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=42)
    freeze.add_argument("--val-fraction", type=float, default=0.34)
    freeze.add_argument("--min-train-scenes", type=int, default=2)
    freeze.add_argument("--min-val-scenes", type=int, default=2)
    freeze.add_argument("--required-task", action="append")
    freeze.add_argument("--prior-manifest", type=Path, action="append", default=[])
    build = sub.add_parser("build", help="admit only the samples in an existing frozen plan")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="revalidate hashes, candidate quality, and frozen scene roles")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--role", choices=("train", "val"))
    verify.add_argument("--for-training", action="store_true")
    for command in (freeze, build, verify):
        command.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    validator = lambda path: default_validator(path, artifact_root=args.artifact_root)
    if args.command == "freeze":
        value = freeze_scene_plan(sample_paths=args.sample, output=args.output,
            parent_checkpoint=args.parent_checkpoint, collection_protocol=args.collection_protocol,
            validator=validator, seed=args.seed, val_fraction=args.val_fraction,
            min_train_scenes=args.min_train_scenes, min_val_scenes=args.min_val_scenes,
            required_tasks=args.required_task or DEFAULT_TASKS, prior_manifests=args.prior_manifest)
    elif args.command == "build":
        value = build_manifest(plan_path=args.plan, output=args.output, validator=validator)
    else:
        value, paths = verify_manifest(args.manifest, validator=validator, role=args.role, for_training=args.for_training)
        print(json.dumps({"scope": SCOPE, "status": "verified", "coverage": value["coverage"],
                          "selected_role": args.role, "selected_paths": [str(path) for path in paths]}, indent=2))
        return 0
    print(json.dumps({"scope": SCOPE, "status": value["status"], "coverage": value["coverage"],
                      "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
