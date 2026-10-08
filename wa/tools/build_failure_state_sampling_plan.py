"""Construct an exclusive, non-training three-source candidate after real admission."""
import argparse
import json
from pathlib import Path

from wa.wm.robot_data import RobotWorldData
from wa.wm.dual_teacher_data import DualTeacherData
from wa.wm.failure_state_data import FailureStateData
from wa.wm.teacher_plan_runtime import load_candidate as load_old_candidate
from wa.wm.failure_state_sampling_runtime import prepare_candidate, save_candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-cache", "base-index", "teacher-cache", "old-plan-root", "old-plan-report-sha",
                 "old-run-root", "student-rows", "teacher-selections", "recovery-cache", "dedup-report",
                 "old-plan-sha", "old-actual-exposure-sha", "recovery-admission-sha", "dedup-report-sha",
                 "project-root", "output"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    print(json.dumps(dict(stage="LOAD_ORIGINAL_SOURCES", training_released=False)),flush=True)
    base = RobotWorldData(a.base_cache,"train",a.base_index)
    teacher = DualTeacherData(a.teacher_cache)
    old_mix,_,_,_ = load_old_candidate(a.old_plan_root,a.old_plan_report_sha,base,teacher,
        base_index=a.base_index,student_rows=a.student_rows,teacher_selections=a.teacher_selections,
        project_root=a.project_root)
    print(json.dumps(dict(stage="LOAD_FAILURE_STATE_ADMISSION", training_released=False)),flush=True)
    recovery = FailureStateData(a.recovery_cache,expected_admission_sha256=a.recovery_admission_sha)
    pins = dict(old_plan=a.old_plan_sha,old_actual_exposure=a.old_actual_exposure_sha,
                recovery_admission=a.recovery_admission_sha,dedup_report=a.dedup_report_sha)
    plan, provenance = prepare_candidate(old_mix,recovery,old_plan_root=a.old_plan_root,
        old_run_root=a.old_run_root,dedup_path=a.dedup_report,source_pins=pins)
    print(json.dumps(dict(stage="EXACT_DDP_SIMULATION_AND_SAVE",training_released=False,
        positions=plan["metadata"]["total_positions"],recovery_budget=plan["metadata"]["budget"])),flush=True)
    result = save_candidate(a.output,plan,provenance)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
