import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_oracle_modular.py"


def write_dataset(repo, task, split, count):
    path = repo / "data" / "datasets" / "track" / task.upper() / split / f"{split}.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as handle:
        json.dump({"episodes": [{} for _ in range(count)]}, handle)


def write_result(root, task, split, index):
    path = root / task / split / "episodes" / f"episode_{index}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "task": task,
        "split": split,
        "controller_version": 5,
        "dataset_index": index,
        "episode_key": f"{task}_{index}",
        "summary": {
            "success": 1.0,
            "following_rate": 0.9,
            "collision": 0.0,
            "finish": True,
            "total_step": 10,
        },
    }))
    return path


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_writes_aggregate_and_task_split_summaries(tmp_path):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    write_dataset(repo, "at", "val", 1)
    write_dataset(repo, "dt", "val", 1)
    write_result(output, "at", "val", 0)
    write_result(output, "dt", "val", 0)

    subprocess.run(
        [sys.executable, str(SCRIPT), str(output), "--repo-root", str(repo)],
        check=True,
    )

    aggregate_rows = read_csv(output / "episodes.csv")
    assert {(row["task"], row["split"]) for row in aggregate_rows} == {
        ("at", "val"),
        ("dt", "val"),
    }
    at_rows = read_csv(output / "at" / "val" / "episodes.csv")
    dt_rows = read_csv(output / "dt" / "val" / "episodes.csv")
    at_summary = json.loads((output / "at" / "val" / "summary.json").read_text())
    dt_summary = json.loads((output / "dt" / "val" / "summary.json").read_text())
    assert [row["task"] for row in at_rows] == ["at"]
    assert [row["task"] for row in dt_rows] == ["dt"]
    assert at_summary["completed"] == 1
    assert dt_summary["completed"] == 1


def test_removes_invalid_json_and_keeps_valid_results(tmp_path):
    output = tmp_path / "output"
    valid_path = write_result(output, "stt", "val", 0)
    invalid_path = output / "stt" / "val" / "episodes" / "truncated.json"
    invalid_path.write_text("")

    subprocess.run(
        [sys.executable, str(SCRIPT), str(output), "--clean-invalid-only"],
        check=True,
    )

    assert valid_path.exists()
    assert not invalid_path.exists()
