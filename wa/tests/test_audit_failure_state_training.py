"""Independent CPU contract tests; never evidence of a real training run."""
import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from wa.tools import audit_failure_state_training as audit
from wa.wm.failure_state_sampling import build_plan, simulate_exposure
from wa.wm.failure_state_sampling_runtime import ThreeSourceMix, RuntimeExposure


def metrics_fixture():
    return dict(status="OFFLINE_ONLY", kind="jepa", steps=61252, closed_loop=False,
        edge_latency_verified=False, point_source="simulated_uwb", elapsed_s=120.,
        metrics={mode: dict(ADE_m=.2, FDE_m=.3, yaw_MAE_rad=.1,
            windows=73368, SR=None, collision_rate=None)
            for mode in ("image", "point", "mixed")})


def lr_multiplier(update):
    # Independent expression matching the frozen trainer, not the auditor helper.
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (update - 1) / 38545))


def logs_fixture():
    base = [2.5e-7, 5e-7, 5e-6, 1e-6, 1e-5]
    rows = []
    for step in range(22708, 61253):
        if step != 22708 and step % 25:
            continue
        rows.append(dict(step=step, total_steps=61252, phase_step=step-22707,
            epoch=1, loss=.92, flow=.8, geometry=.2, world=.2, grad_norm=.4,
            elapsed_s=float(step-22707), peak_allocated_gib=8.,
            loss_aggregation="DDP_mean_over_accumulation_group",
            loss_rank0_last_microbatch=.93,
            lr=[x*lr_multiplier(step-22707) for x in base]))
    return rows, base


def checkpoints_fixture():
    base = [2.5e-7, 5e-7, 5e-6, 1e-6, 1e-5]
    parent = dict(step=22707, completed_epochs=1, kind="jepa", contract=audit.CONTRACT,
        model={f"head.{i}": torch.ones(2) for i in range(5)},
        optimizer=dict(param_groups=[dict(params=[i], lr=base[i], weight_decay=.01,
            betas=(.9,.999), eps=1e-8, amsgrad=False) for i in range(5)],
            state={i: dict(step=torch.tensor(22707.), exp_avg=torch.ones(2)*.1,
                exp_avg_sq=torch.ones(2)*.2) for i in range(5)}))
    final = copy.deepcopy(parent)
    final.update(step=61252, completed_epochs=2, parent_checkpoint=audit.PARENT_PATH,
                 parent_sha256=audit.PARENT_SHA)
    for group in final["optimizer"]["param_groups"]:
        group["lr"] *= lr_multiplier(38545)
    for state in final["optimizer"]["state"].values():
        state["step"] = torch.tensor(61252.)
        state["exp_avg"] += .01
    for value in final["model"].values():
        value += .01
    return parent, final, dict(base_lrs=base)


def exposure_fixture():
    old = [("base", i) for i in range(8)] + [("teacher", i % 4) for i in range(8)]
    old.insert(3, ("teacher", 7))
    consumed = [1]*17
    consumed[3] = 0
    windows = []
    for i in range(4):
        windows.append(dict(dataset_index=100+2*i, raw_row=1000+i,
            episode=f"stt:scene/{i//2}", task="stt", key=f"scene/{i//2}",
            teacher="oracle", current_index=20+i%2, takeover_step=10,
            current_age_s=.5 if i%2==0 else 1.5, valid=True, exact_duplicate=False))
    pins = {name: str(i)*64 for i,name in enumerate(
        ("old_plan","old_actual_exposure","recovery_admission","dedup_report"),1)}
    plan = build_plan(old, consumed, windows, source_pins=pins,
                      recovery_budget=16, expected_dropped_positions=(3,))
    oldmix = SimpleNamespace(base=range(8), teacher=range(8))
    mix = ThreeSourceMix(plan, oldmix, range(107))
    counter = RuntimeExposure(mix)
    for rank in range(8):
        sampler = DistributedSampler(mix, num_replicas=8, rank=rank,
                                     seed=42, drop_last=True)
        sampler.set_epoch(1)
        positions = list(sampler)
        counter.consume(torch.tensor(positions[:len(positions)//2*2],dtype=torch.int64))
    state = counter.state()
    actual = counter.finalize(state=state,world=8,batch=2,seed=42,epoch=1)
    actual.update(optimizer_steps_completed=38545,diagnostic=False)
    arrays = dict(position_counts=state["position_counts"].numpy(),
                  **{name: value.numpy() for name,value in zip(
                      ("base_counts","teacher_counts","recovery_counts"),
                      counter.source_counts(state))})
    old_arrays = dict(position_counts=np.asarray(consumed,dtype=np.int64),
        base_counts=np.ones(8,dtype=np.int64),
        teacher_counts=np.array([2,2,2,2,0,0,0,0],dtype=np.int64))
    return plan, actual, arrays, old_arrays, simulate_exposure(plan)


class FailureStateTrainingAuditTests(unittest.TestCase):
    def test_formal_constants_are_not_test_rescaled(self):
        self.assertEqual((audit.START,audit.FINAL,audit.UPDATES,audit.HELDOUT),
                         (22707,61252,38545,73368))
        self.assertEqual(audit.SOURCE_COMMIT,"199385cd9c826c8f21308ad99b6a8375396d9c90")

    def test_missing_metrics_never_loads_checkpoint(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"checkpoint.pt").write_bytes(b"unfinished checkpoint")
            loader=Mock(side_effect=AssertionError("must not load unfinished checkpoint"))
            with self.assertRaises(audit.IncompleteTraining):
                audit.audit_training(root,root,root/"job.yaml",root,
                    source_commit=audit.SOURCE_COMMIT,config_sha256="a"*64,
                    plan_admission_sha256="b"*64,checkpoint_loader=loader)
            loader.assert_not_called()

    def test_metrics_complete_three_modes(self):
        audit.validate_metrics(metrics_fixture())

    def test_metrics_wrong_identity_and_types(self):
        for field,value in (("steps",61251),("steps",61252.),("closed_loop",True),
            ("closed_loop",0),("kind","dino"),("status","DIAGNOSTIC_PASS"),
            ("point_source","future_gt"),("edge_latency_verified",True)):
            bad=metrics_fixture();bad[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                audit.validate_metrics(bad)

    def test_metrics_modes_and_heldout_are_exact(self):
        for mode in ("image","point","mixed"):
            bad=metrics_fixture();del bad["metrics"][mode]
            with self.subTest(missing=mode),self.assertRaises(ValueError):
                audit.validate_metrics(bad)
            for field,value in (("windows",16),("windows",73368.),
                ("ADE_m",float("nan")),("FDE_m",float("inf")),
                ("yaw_MAE_rad",-1.),("SR",.8),("collision_rate",0.),("ADE_m",True)):
                bad=metrics_fixture();bad["metrics"][mode][field]=value
                with self.subTest(mode=mode,field=field),self.assertRaises(ValueError):
                    audit.validate_metrics(bad)
        bad=metrics_fixture();bad["metrics"]["extra"]=copy.deepcopy(bad["metrics"]["mixed"])
        with self.assertRaises(ValueError):audit.validate_metrics(bad)

    def test_all_expected_sparse_logs_and_resumed_lr(self):
        rows,base=logs_fixture()
        self.assertEqual(len(rows),1543)
        self.assertEqual((rows[0]["step"],rows[-1]["step"]),(22708,61250))
        audit.validate_logs(rows,base)

    def test_logs_no_missing_duplicate_reordering_or_fabricated_tail(self):
        rows,base=logs_fixture()
        for bad in (rows[1:],rows[:-1],rows[:100]+rows[101:],rows[::-1],
                    rows+[rows[-1]],rows+[dict(rows[-1],step=61252,phase_step=38545)]):
            with self.subTest(length=len(bad)),self.assertRaises(ValueError):
                audit.validate_logs(bad,base)

    def test_logs_loss_lr_and_nonfinite_rejected(self):
        rows,base=logs_fixture()
        for field,value in (("total_steps",61251),("epoch",2),("phase_step",0),
            ("loss",1.),("world",float("nan")),("grad_norm",float("inf")),
            ("loss_aggregation","rank0"),("elapsed_s",float("nan"))):
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_logs(bad,base)
        bad=copy.deepcopy(rows);bad[0]["lr"][0]*=10
        with self.assertRaises(ValueError):audit.validate_logs(bad,base)
        bad=copy.deepcopy(rows);bad[-1]["lr"]=base
        with self.assertRaises(ValueError):audit.validate_logs(bad,base)

    def test_parent_and_optimizer_positive(self):
        audit.validate_parent_continuation(*checkpoints_fixture())

    def test_optimizer_reset_missing_state_and_group_mapping(self):
        for mutation in ("reset","drop_state","group_count","params","weight_decay","lr","base_lr"):
            before,after,config=checkpoints_fixture()
            if mutation=="reset":after["optimizer"]["state"][0]["step"]=torch.tensor(1.)
            elif mutation=="drop_state":del after["optimizer"]["state"][0]
            elif mutation=="group_count":after["optimizer"]["param_groups"].pop()
            elif mutation=="params":after["optimizer"]["param_groups"][0]["params"]=[99]
            elif mutation=="weight_decay":after["optimizer"]["param_groups"][0]["weight_decay"]=.1
            elif mutation=="lr":after["optimizer"]["param_groups"][0]["lr"]=1e-3
            else:config["base_lrs"][0]*=10
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                audit.validate_parent_continuation(before,after,config)

    def test_optimizer_parent_identity_model_layout_and_state_values(self):
        for field,value in (("step",61251),("completed_epochs",3),
                            ("parent_sha256","a"*64),("parent_checkpoint","/wrong.pt")):
            before,after,config=checkpoints_fixture();after[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_parent_continuation(before,after,config)
        for change in ("model_shape","model_dtype","parent_step","nonfinite_step","too_many_steps"):
            before,after,config=checkpoints_fixture()
            if change=="model_shape":after["model"]["head.0"]=torch.zeros(3)
            elif change=="model_dtype":after["model"]["head.0"]=after["model"]["head.0"].double()
            elif change=="parent_step":before["step"]=22706
            elif change=="nonfinite_step":after["optimizer"]["state"][0]["step"]=torch.tensor(float("nan"))
            else:after["optimizer"]["state"][0]["step"]=torch.tensor(61253.)
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_parent_continuation(before,after,config)

    def test_real_small_three_source_exposure_positive(self):
        result=audit.validate_exposure(*exposure_fixture())
        self.assertIn("calculated",result)
        self.assertIn("ranks",result)

    def test_four_npz_arrays_required_with_strict_dtype_and_shape(self):
        for name in ("position_counts","base_counts","teacher_counts","recovery_counts"):
            for change in ("missing","dtype","shape","negative"):
                fixture=list(exposure_fixture())
                if change=="missing":del fixture[2][name]
                elif change=="dtype":fixture[2][name]=fixture[2][name].astype(np.float64)
                elif change=="shape":fixture[2][name]=fixture[2][name][:-1]
                else:fixture[2][name][0]=-1
                with self.subTest(name=name,change=change),self.assertRaises(ValueError):
                    audit.validate_exposure(*fixture)
        fixture=list(exposure_fixture());fixture[2]["unexpected"]=np.zeros(1,dtype=np.int64)
        with self.assertRaises(ValueError):audit.validate_exposure(*fixture)

    def test_same_total_per_window_swaps_and_missing_repeat_rejected(self):
        for name,pair in (("position_counts",(0,1)),("base_counts",(0,1)),
                          ("teacher_counts",(0,1)),("recovery_counts",(100,102))):
            fixture=list(exposure_fixture());a=fixture[2][name];total=int(a.sum())
            a[pair[0]]-=1;a[pair[1]]+=1
            self.assertEqual(int(a.sum()),total)
            with self.subTest(name=name),self.assertRaises(ValueError):
                audit.validate_exposure(*fixture)

    def test_old_source_exact_per_window_exposure_not_just_totals(self):
        fixture=list(exposure_fixture());a=fixture[3]["teacher_counts"]
        a[0]-=1;a[1]+=1
        self.assertEqual(int(a.sum()),8)
        with self.assertRaises(ValueError):audit.validate_exposure(*fixture)

    def test_exposure_json_seed_epoch_plan_or_sampling_disagreement(self):
        for field,value in (("diagnostic",True),("optimizer_steps_completed",1),
            ("plan_sha256","a"*64),("actual_recovery",15),("world_size",1),("seed",43)):
            fixture=list(exposure_fixture());fixture[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_exposure(*fixture)
        fixture=list(exposure_fixture());fixture[4]["seed"]=43
        with self.assertRaises(ValueError):audit.validate_exposure(*fixture)


    def test_optimizer_nonfinite_weights_and_adam_moments_rejected(self):
        for container,name,value in (("model","head.0",float("nan")),
            ("model","head.1",float("inf")),("state","exp_avg",float("nan")),
            ("state","exp_avg_sq",float("inf"))):
            before,after,config=checkpoints_fixture()
            if container=="model":after["model"][name][0]=value
            else:after["optimizer"]["state"][0][name][0]=value
            with self.subTest(container=container,name=name),self.assertRaises(ValueError):
                audit.validate_parent_continuation(before,after,config)

    def environment_fixture(self):
        root=Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
        hashes={"wa/wm/train.py":"a"*64,"wa/wm/training.py":"b"*64}
        selection=root/"artifacts/failure_state_group_fit_selection_20261008_v1.json"
        fit=root/"artifacts/failure_state_group_fit_pretrain_20261008_v1.json"
        selection_sha="7e7db844c09f67d7b0e3b5de7376c0573c18fccc485dfa235b879ed4dab111a9"
        fit_sha="6ecff4bda1c19d153e7e4be0742914d2b761698759a9157c17ec2920a6725cd9"
        wla=dict(checkpoint_sha256="0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1",
            checkpoint_step=43203,strict_loaded=["action_expert","metaquery","target_head"],
            omitted=["Qwen backbone","language LoRA"],
            action_contract="7x4 XY/sin(yaw)/cos(yaw)",source_files={"src.py":"c"*64})
        env=dict(commit=audit.SOURCE_COMMIT,dirty="",torch="2.8.0+cu128",
            gpu_names=["NVIDIA GeForce RTX 4090"]*8,source_sha256=hashes,wla=wla)
        evidence=dict(selection_path=str(selection),selection_sha256=selection_sha,
            fit_path=str(fit),fit_sha256=fit_sha,schema="failure_state_group_fit_v1",
            status="FIXED152_TEACHER_STATE_LABEL_FIT_ONLY_NOT_SR",windows=152,
            window_predictions=608,models={
                "parent59866":dict(sha256=audit.PARENT_SHA,step=22707),
                "best61609":dict(sha256="c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52",step=59716)},
            policy_forward_only=True,full_training_recipe_4090_verified=False,
            closed_loop=False,training_released=False)
        launch=dict(status="WORKER_PREFLIGHT_PASSED_NOT_TRAINING_COMPLETE",
            source_commit=audit.SOURCE_COMMIT,
            cuda=[dict(index=i,name="NVIDIA GeForce RTX 4090",capability=[8,9]) for i in range(8)],
            uuid_model_memory_driver=[[f"GPU-00000000-0000-0000-0000-{i:012d}",
                "NVIDIA GeForce RTX 4090","24564","570.124.06"] for i in range(8)],
            input_sha256={str(selection):selection_sha,str(fit):fit_sha,
                          audit.PARENT_PATH:audit.PARENT_SHA},
            fixed_group_fit_evidence=evidence,wla_source_files=1,output="/run",torch="2.8.0+cu128")
        return env,hashes,launch,audit.SOURCE_COMMIT

    def test_actual_environment_and_eight_unique_gpu_uuids(self):
        audit.validate_environment(*self.environment_fixture())

    def test_gpu_duplicate_missing_wrong_model_memory_and_device_mapping(self):
        for change in ("duplicate_uuid","seven_devices","seven_uuid","wrong_model",
                       "small_memory","wrong_capability","duplicate_index"):
            env,hashes,launch,commit=self.environment_fixture()
            if change=="duplicate_uuid":launch["uuid_model_memory_driver"][1][0]=launch["uuid_model_memory_driver"][0][0]
            elif change=="seven_devices":launch["cuda"].pop()
            elif change=="seven_uuid":launch["uuid_model_memory_driver"].pop()
            elif change=="wrong_model":launch["uuid_model_memory_driver"][0][1]="NVIDIA H100"
            elif change=="small_memory":launch["uuid_model_memory_driver"][0][2]="10000"
            elif change=="wrong_capability":launch["cuda"][0]["capability"]=[8,0]
            else:launch["cuda"][1]["index"]=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_environment(env,hashes,launch,commit)

    def test_worker_source_framework_dirty_and_gpu_names_rejected(self):
        for field,value in (("commit","d"*40),("dirty"," M wa/wm/train.py"),
            ("torch","2.7.0"),("gpu_names",["NVIDIA GeForce RTX 4090"]*7),
            ("source_sha256",{}),("gpu_names",["NVIDIA A800"]*8)):
            env,hashes,launch,commit=self.environment_fixture();env[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_environment(env,hashes,launch,commit)
        env,hashes,launch,commit=self.environment_fixture()
        hashes=dict(hashes);hashes["wa/wm/train.py"]="f"*64
        with self.assertRaises(ValueError):audit.validate_environment(env,hashes,launch,commit)

    def test_yaml_command_known_variables_expand_and_flag_types(self):
        env={"WA_ROOT":"/nas/project","WA_PARENT_CHECKPOINT":"/nas/parent.pt"}
        cmd=('bash wa/scripts/train_world.sh --epochs 1 --resume "$WA_PARENT_CHECKPOINT" '
             '--failure-state-cache "$WA_ROOT/cache" --evaluation-set-adaptation')
        got=audit.parse_training_command(cmd,env)
        self.assertEqual(got,{"--epochs":"1","--resume":"/nas/parent.pt",
            "--failure-state-cache":"/nas/project/cache","--evaluation-set-adaptation":True})

    def test_yaml_duplicate_flag_unknown_variable_or_extra_launcher_rejected(self):
        env={"WA_ROOT":"/nas/project","WA_PARENT_CHECKPOINT":"/nas/parent.pt"}
        start="bash wa/scripts/train_world.sh "
        for cmd in (start+"--epochs 1 --epochs 2",
                    start+'--resume "$UNKNOWN_ROOT/parent.pt"',
                    start+'--resume "'+chr(36)+'{UNKNOWN_ROOT}/parent.pt"',
                    start+"--epochs",
                    start+"--epochs 1\n"+start+"--epochs 1"):
            with self.subTest(cmd=cmd),self.assertRaises(ValueError):
                audit.parse_training_command(cmd,env)



    def config_fixture(self):
        run=audit.JOBS/"job_99999/task_88888/wa_failure_state_train_4090_v1"
        source=audit.ROOT/"source_failure_state_train_v1"
        config=audit.expected_recipe(run,audit.PLAN_ROOT)
        config["base_lrs"]=[2.5e-7,5e-7,5e-6,1e-6,1e-5]
        exports=dict(WA_ROOT=str(audit.ROOT),WA_WLA_SOURCE=config["wla_source"],
            WA_WLA_CHECKPOINT=config["wla_checkpoint"],WA_ENCODER_WEIGHTS=config["encoder_weight"],
            WA_PARENT_CHECKPOINT=audit.PARENT_PATH,WA_CACHE=str(audit.BASE),
            WA_INDEX_ROOT=str(audit.INDEX),WA_PYTHON=str(audit.ROOT/"probe_env/bin/python"),
            WA_WORLD_KIND="jepa",WA_GPUS="8",WA_OUTPUT=str(run),OMP_NUM_THREADS="2",
            NCCL_DEBUG="WARN",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")
        # These are command bytes, never executed. Only explicit named exports are legal.
        command=["set -euo pipefail","cd "+str(source),"git status --porcelain",
                 "# frozen "+audit.SOURCE_COMMIT]
        command.extend(f'export {key}="{value}"' for key,value in exports.items())
        flags=[]
        for name in sorted(audit.FLAGS):
            flags.append("--"+name)
            if name!="evaluation-set-adaptation":
                flags.append(str(config[name.replace("-","_")]))
        command.append("bash wa/scripts/train_world.sh "+" ".join(flags))
        job=dict(cluster="baidu_4090",num_gpus=8,
            image="x5-builder:cuda12.8-isaac5.0.0-v2.test1",
            tasks=[dict(num_gpus=8,type="shell",workload_backend="k8s",timeout=86400,
                        cmd="\n".join(command))])
        return config,job,source,run,audit.PLAN_ROOT,audit.SOURCE_COMMIT

    def test_full_yaml_runtime_config_matches_fixed_one_epoch_recipe(self):
        fixture=self.config_fixture()
        config=fixture[0]
        self.assertEqual((config["train_rows"],config["heldout_rows"],
            config["world_size"],config["effective_batch"],config["epochs"],
            config["completed_epochs"]),(1233424,73368,8,32,1,1))
        got=audit.validate_config(*fixture)
        self.assertEqual(len(got["arguments"]),26)
        self.assertEqual(got["arguments"]["--resume"],audit.PARENT_PATH)
        self.assertEqual(got["arguments"]["--failure-state-plan-admission-sha256"],
                         audit.PLAN_ADMISSION_SHA)

    def test_config_new_recipe_pins_optimizer_source_and_legacy_routes_rejected(self):
        mutations=(("world_size",1),("effective_batch",16),("train_rows",1233423),
            ("heldout_rows",73367),("epochs",2),("completed_epochs",0),("seed",7),
            ("diagnostic",True),("evaluation_set_adaptation",False),("kind","dino"),
            ("resume","/new.pt"),("resume_sha256","f"*64),("failure_state_admission_sha256","f"*64),
            ("failure_state_plan_admission_sha256","f"*64),
            ("failure_state_old_plan_sha256","f"*64),("failure_state_old_exposure_sha256","f"*64),
            ("failure_state_dedup_sha256","f"*64),("failure_state_loader_audit_sha256","f"*64),
            ("dual_teacher_repeats",2),("world_weight",.2),
            ("history_repeat_probability",.5),("recovery_cache","/foreign/cache"),
            ("recovery_index","/foreign/index"),("data_root","/foreign/data"),
            ("source_prefix","/foreign/root"),("base_lrs",[1e-5]*4),
            ("base_lrs",[float("nan")]*5),("epochs",True))
        for field,value in mutations:
            fixture=list(self.config_fixture());fixture[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_config(*fixture)

    def test_yaml_hardware_timeout_source_and_output_guard_rejected(self):
        for change in ("cluster","gpu","task_gpu","image","timeout","two_tasks",
                       "source_commit","missing_cd","missing_clean_guard","output_route"):
            fixture=list(self.config_fixture());job=fixture[1];task=job["tasks"][0]
            if change=="cluster":job["cluster"]="baidu_a800"
            elif change=="gpu":job["num_gpus"]=7
            elif change=="task_gpu":task["num_gpus"]=7
            elif change=="image":job["image"]="unverified:latest"
            elif change=="timeout":task["timeout"]=1800
            elif change=="two_tasks":job["tasks"].append(copy.deepcopy(task))
            elif change=="source_commit":fixture[-1]="f"*40
            elif change=="missing_cd":task["cmd"]=task["cmd"].replace("cd ","# cwd ",1)
            elif change=="missing_clean_guard":task["cmd"]=task["cmd"].replace("git status --porcelain","true")
            else:
                fixture[3]=audit.JOBS/"job_99999/task_88888/other"
                fixture[0]["output"]=str(fixture[3])
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_config(*fixture)

    def test_full_yaml_duplicate_unknown_export_or_training_flags_rejected(self):
        for change in ("duplicate_export","unknown_export","unknown_variable",
                       "duplicate_flag","missing_flag","wrong_flag","unknown_flag"):
            fixture=list(self.config_fixture());task=fixture[1]["tasks"][0]
            if change=="duplicate_export":task["cmd"]+="\nexport WA_GPUS=8"
            elif change=="unknown_export":task["cmd"]+="\nexport WA_UNKNOWN=1"
            elif change=="unknown_variable":task["cmd"]=task["cmd"].replace(
                "--epochs 1","--epochs "+chr(36)+"{UNKNOWN_EPOCHS}")
            elif change=="duplicate_flag":task["cmd"]+=" --epochs 1"
            elif change=="missing_flag":task["cmd"]=task["cmd"].replace("--epochs 1 ","")
            elif change=="wrong_flag":task["cmd"]=task["cmd"].replace("--epochs 1 ","--epochs 2 ")
            else:task["cmd"]+=" --unapproved-option 1"
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_config(*fixture)

    def test_incomplete_or_invalid_metrics_never_reaches_checkpoint_loader(self):
        for change in ("partial_json","duplicate_json","wrong_step","missing_mode","short_heldout"):
            with TemporaryDirectory() as directory:
                root=Path(directory)
                for name in audit.TERMINAL:
                    (root/name).write_bytes(b"present but not valid for downstream use")
                metrics=metrics_fixture()
                if change=="wrong_step":metrics["steps"]=61251
                elif change=="missing_mode":del metrics["metrics"]["point"]
                elif change=="short_heldout":metrics["metrics"]["mixed"]["windows"]=73367
                blob=json.dumps(metrics)
                if change=="partial_json":blob=blob[:-10]
                elif change=="duplicate_json":blob=blob.replace(
                    '"steps": 61252','"steps": 61252, "steps": 61252')
                (root/"metrics.json").write_text(blob)
                loader=Mock(side_effect=AssertionError("early metrics gate was bypassed"))
                with self.subTest(change=change),self.assertRaises(ValueError) as caught:
                    audit.audit_training(root,root,root/"job.yaml",root,
                        source_commit=audit.SOURCE_COMMIT,config_sha256="a"*64,
                        plan_admission_sha256=audit.PLAN_ADMISSION_SHA,checkpoint_loader=loader)
                self.assertNotIn("fixed source/plan",str(caught.exception))
                loader.assert_not_called()

    def test_terminal_metrics_positive_missing_file_and_symlink_fail_closed(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for name in audit.TERMINAL:(root/name).write_bytes(b"present")
            (root/"metrics.json").write_text(json.dumps(metrics_fixture()))
            self.assertEqual(audit.terminal_metrics(root)["steps"],61252)
            (root/"checkpoint.pt").unlink()
            with self.assertRaises(audit.IncompleteTraining):audit.terminal_metrics(root)
            (root/"checkpoint.pt").symlink_to(root/"config.json")
            with self.assertRaises(audit.IncompleteTraining):audit.terminal_metrics(root)


    def test_optimizer_cannot_fake_progress_with_unchanged_model(self):
        before,after,config=checkpoints_fixture()
        after["model"]=copy.deepcopy(before["model"])
        self.assertEqual(after["step"],61252)
        self.assertEqual(max(int(s["step"]) for s in after["optimizer"]["state"].values()),61252)
        with self.assertRaisesRegex(ValueError,"all nonencoder model tensors unchanged"):
            audit.validate_parent_continuation(before,after,config)

    def test_adam_moment_shape_dtype_and_negative_square_are_rejected(self):
        for change in ("paired_shape","paired_dtype","internal_shape","internal_dtype",
                       "negative_parent_square","negative_final_square"):
            before,after,config=checkpoints_fixture()
            a=before["optimizer"]["state"][0];b=after["optimizer"]["state"][0]
            if change=="paired_shape":
                b["exp_avg"]=torch.ones(3);b["exp_avg_sq"]=torch.ones(3)
            elif change=="paired_dtype":
                b["exp_avg"]=b["exp_avg"].double();b["exp_avg_sq"]=b["exp_avg_sq"].double()
            elif change=="internal_shape":b["exp_avg"]=torch.ones(3)
            elif change=="internal_dtype":b["exp_avg"]=b["exp_avg"].double()
            elif change=="negative_parent_square":a["exp_avg_sq"][0]=-.1
            else:b["exp_avg_sq"][0]=-.1
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_parent_continuation(before,after,config)


    def test_full_active_parameter_cannot_stop_or_partly_advance(self):
        # Other states still reach FINAL: max(step) alone must not certify this one.
        for stopped_at in (22707,22708,61251):
            before,after,config=checkpoints_fixture()
            after["optimizer"]["state"][0]["step"]=torch.tensor(float(stopped_at))
            self.assertEqual(max(int(s["step"]) for s in after["optimizer"]["state"].values()),61252)
            with self.subTest(stopped_at=stopped_at),self.assertRaises(ValueError):
                audit.validate_parent_continuation(before,after,config)

    def test_full_active_optimizer_group_cannot_be_retained_from_parent(self):
        before,after,config=checkpoints_fixture()
        # A two-parameter group is wholly stale; four other groups finish normally.
        for checkpoint in (before,after):
            checkpoint["model"]["head.5"]=checkpoint["model"]["head.0"].clone()
            checkpoint["optimizer"]["param_groups"][0]["params"].append(5)
            checkpoint["optimizer"]["state"][5]=copy.deepcopy(checkpoint["optimizer"]["state"][0])
        for key in (0,5):
            after["optimizer"]["state"][key]=copy.deepcopy(before["optimizer"]["state"][key])
        self.assertEqual(max(int(s["step"]) for s in after["optimizer"]["state"].values()),61252)
        with self.assertRaises(ValueError):
            audit.validate_parent_continuation(before,after,config)

    def test_conditionally_active_parent_state_keeps_bounded_compatibility(self):
        result=audit.validate_parent_continuation(*checkpoints_fixture())
        self.assertEqual(result["planned_full_continuation_state_count"],5)
        self.assertEqual(result["conditional_parent_state_count"],0)
        self.assertEqual(result["new_optimizer_state_count"],0)
        for new_step in (22706,22707,61251):
            before,after,config=checkpoints_fixture()
            before["optimizer"]["state"][0]["step"]=torch.tensor(22706.)
            after["optimizer"]["state"][0]["step"]=torch.tensor(float(new_step))
            result=audit.validate_parent_continuation(before,after,config)
            with self.subTest(new_step=new_step):
                self.assertEqual(result["planned_full_continuation_state_count"],4)
                self.assertEqual(result["conditional_parent_state_count"],1)
                self.assertIsInstance(result["conditional_state_scope"],str)
                self.assertTrue(result["conditional_state_scope"])
        before,after,config=checkpoints_fixture()
        before["optimizer"]["state"][0]["step"]=torch.tensor(22706.)
        after["optimizer"]["state"][0]["step"]=torch.tensor(61252.)
        with self.assertRaises(ValueError):
            audit.validate_parent_continuation(before,after,config)


    def test_metrics_replaced_between_first_read_and_pinning_fail_before_loader(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            for name in audit.TERMINAL:
                (root/name).write_bytes(b"present small terminal fixture")
            metrics_path=root/"metrics.json"
            original=metrics_fixture()
            metrics_path.write_text(json.dumps(original))
            original_sha=audit.digest(metrics_path)
            replacement=metrics_fixture()
            replacement["metrics"]["mixed"]["ADE_m"]=.25
            # Both versions independently satisfy metric contracts; identity must bind.
            audit.validate_metrics(replacement)
            read_terminal=audit.terminal_metrics
            real_add=audit.Pins.add
            recorded={}
            def swap_after_validated_first_read(run):
                first=read_terminal(run)
                self.assertEqual(first,original)
                metrics_path.write_text(json.dumps(replacement))
                return first
            def real_pin_and_record(pins,path,expected=None,**kwargs):
                value=real_add(pins,path,expected,**kwargs)
                recorded[str(path)]=value
                return value
            def only_source_git(_root,*arguments):
                if arguments==("rev-parse","HEAD"):return audit.SOURCE_COMMIT
                if arguments==("status","--porcelain"):return ""
                raise AssertionError("unexpected downstream git call")
            loader=Mock(side_effect=AssertionError("checkpoint loader must remain unreachable"))
            with patch.object(audit,"terminal_metrics",side_effect=swap_after_validated_first_read), \
                 patch.object(audit,"git",side_effect=only_source_git), \
                 patch.object(audit.Pins,"add",new=real_pin_and_record):
                with self.assertRaisesRegex(ValueError,
                        "final metrics changed between completion gate and file pinning"):
                    audit.audit_training(root,audit.ROOT/"source_failure_state_train_v1",
                        root/"not_reached.yaml",audit.PLAN_ROOT,
                        source_commit=audit.SOURCE_COMMIT,config_sha256="a"*64,
                        plan_admission_sha256=audit.PLAN_ADMISSION_SHA,
                        checkpoint_loader=loader)
            loader.assert_not_called()
            self.assertEqual(recorded[str(metrics_path)],audit.digest(metrics_path))
            self.assertNotEqual(recorded[str(metrics_path)],original_sha)
            self.assertEqual(set(recorded),
                {str(root/name) for name in audit.TERMINAL if name!="checkpoint.pt"})


    def a800_config_fixture(self):
        fixture=list(self.config_fixture())
        old=str(fixture[3])
        fixture[3]=audit.JOBS/"job_99999/task_88888/wa_failure_state_train_a800_v1"
        fixture[0]["output"]=str(fixture[3])
        fixture[1]["cluster"]="baidu_a800"
        fixture[1]["tasks"][0]["cmd"]=fixture[1]["tasks"][0]["cmd"].replace(old,str(fixture[3]))
        return fixture

    def a800_environment_fixture(self):
        env,hashes,launch,commit=self.environment_fixture()
        launch["hardware_profile"]="a800"
        env["gpu_names"]=["NVIDIA A800-SXM4-80GB"]*8
        for record in launch["cuda"]:
            record.update(name="NVIDIA A800-SXM4-80GB",capability=[8,0])
        for record in launch["uuid_model_memory_driver"]:
            record[1]="NVIDIA A800-SXM4-80GB"
            record[2]="81920"
        return env,hashes,launch,commit

    def test_explicit_a800_config_and_environment_are_accepted(self):
        config=audit.validate_config(*self.a800_config_fixture(),hardware_profile="a800")
        hardware=audit.validate_environment(*self.a800_environment_fixture(),hardware_profile="a800")
        self.assertEqual(config["hardware_profile"],"a800")
        self.assertEqual(hardware["hardware_profile"],"a800")
        self.assertEqual(len(hardware["unique_gpu_uuids"]),8)
        self.assertEqual(audit.hardware_contract("a800")["min_memory_mib"],80000)

    def test_profiles_cannot_cross_accept_cluster_output_or_hardware(self):
        for fixture,profile in ((self.config_fixture(),"a800"),(self.a800_config_fixture(),"rtx4090")):
            with self.subTest(config_profile=profile),self.assertRaises(ValueError):
                audit.validate_config(*fixture,hardware_profile=profile)
        for fixture,profile in ((self.environment_fixture(),"a800"),(self.a800_environment_fixture(),"rtx4090")):
            with self.subTest(hardware_profile=profile),self.assertRaises(ValueError):
                audit.validate_environment(*fixture,hardware_profile=profile)
        # A800 requires explicit selection; the old default remains RTX4090.
        with self.assertRaises(ValueError):audit.validate_config(*self.a800_config_fixture())
        with self.assertRaises(ValueError):audit.validate_environment(*self.a800_environment_fixture())

    def test_a800_mixed_or_incorrect_gpu_evidence_is_rejected(self):
        for change in ("env_mixed","cuda_mixed","physical_mixed","similar_model",
                       "wrong_capability","insufficient_memory","duplicate_uuid","seven_cards"):
            fixture=list(self.a800_environment_fixture());env,_,launch,_=fixture
            if change=="env_mixed":env["gpu_names"][0]="NVIDIA GeForce RTX 4090"
            elif change=="cuda_mixed":launch["cuda"][0]["name"]="NVIDIA GeForce RTX 4090"
            elif change=="physical_mixed":launch["uuid_model_memory_driver"][0][1]="NVIDIA GeForce RTX 4090"
            elif change=="similar_model":launch["uuid_model_memory_driver"][0][1]="NVIDIA A800 80GB PCIe"
            elif change=="wrong_capability":launch["cuda"][0]["capability"]=[8,9]
            elif change=="insufficient_memory":launch["uuid_model_memory_driver"][0][2]="79999"
            elif change=="duplicate_uuid":launch["uuid_model_memory_driver"][1][0]=launch["uuid_model_memory_driver"][0][0]
            else:launch["uuid_model_memory_driver"].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_environment(*fixture,hardware_profile="a800")

    def test_a800_cluster_and_run_route_are_exact(self):
        for change in ("wrong_cluster","rtx_output","generic_output","wrong_gpu_count"):
            fixture=self.a800_config_fixture()
            if change=="wrong_cluster":fixture[1]["cluster"]="baidu_4090"
            elif change=="wrong_gpu_count":fixture[1]["num_gpus"]=7
            else:
                old=str(fixture[3])
                suffix="wa_failure_state_train_4090_v1" if change=="rtx_output" else "wa_train"
                fixture[3]=audit.JOBS/f"job_99999/task_88888/{suffix}"
                fixture[0]["output"]=str(fixture[3])
                fixture[1]["tasks"][0]["cmd"]=fixture[1]["tasks"][0]["cmd"].replace(old,str(fixture[3]))
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_config(*fixture,hardware_profile="a800")

    def test_unknown_hardware_profiles_fail_closed(self):
        for profile in ("A800","h100","auto","",None,False):
            with self.subTest(profile=profile),self.assertRaises(ValueError):
                audit.hardware_contract(profile)


    def a800_developer_fixture(self):
        launch=self.a800_environment_fixture()[2]
        launch["input_sha256"].update(audit.A800_DEVELOPER_PINS)
        launch["developer_training_evidence"]=dict(
            schema="failure_state_developer_training_evidence_v1",
            status="A800_FOUR_UPDATE_DIAGNOSTIC_PASS_NOT_EIGHT_RANK_PROOF",
            hardware_profile="a800",run=str(audit.A800_DEVELOPER_RUN),
            source_commit=audit.SOURCE_COMMIT,input_sha256=dict(audit.A800_DEVELOPER_PINS),
            world_size=1,first_step=22708,final_step=22711,optimizer_updates=4,
            consumed_positions=16,source_counts=dict(base=7,teacher=6,recovery=3),
            heldout_windows_per_mode=2,training_recipe_verified_single_gpu=True,
            eight_rank_gpu_execution_verified=False,closed_loop=False)
        return launch

    def test_a800_developer_evidence_is_fixed_single_gpu_only(self):
        launch=self.a800_developer_fixture()
        result=audit.validate_developer_training_evidence(launch)
        self.assertEqual(len(result["input_sha256"]),7)
        self.assertEqual(result["source_counts"],dict(base=7,teacher=6,recovery=3))
        self.assertIs(result["eight_rank_gpu_execution_verified"],False)
        pins=Mock()
        audit.bind_developer_training(pins,launch,"a800")
        self.assertEqual(pins.add.call_count,7)
        self.assertEqual({call.args for call in pins.add.call_args_list},
                         set(audit.A800_DEVELOPER_PINS.items()))

    def test_a800_developer_evidence_wrong_claims_or_pins_rejected(self):
        for key,value in (("schema","other"),("status","FORMAL_PASS"),("hardware_profile","rtx4090"),
            ("source_commit","f"*40),("run","/other"),("world_size",8),("first_step",22707),
            ("final_step",61252),("optimizer_updates",5),("consumed_positions",1233424),
            ("source_counts",dict(base=8,teacher=5,recovery=3)),("heldout_windows_per_mode",73368),
            ("training_recipe_verified_single_gpu",False),("eight_rank_gpu_execution_verified",True),
            ("closed_loop",True),("closed_loop",0)):
            launch=self.a800_developer_fixture()
            launch["developer_training_evidence"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit.validate_developer_training_evidence(launch)
        for change in ("missing_evidence","missing_evidence_pin","changed_evidence_pin",
                       "extra_evidence_pin","missing_launch_pin","changed_launch_pin","wrong_launch_profile"):
            launch=self.a800_developer_fixture()
            path=next(iter(audit.A800_DEVELOPER_PINS))
            if change=="missing_evidence":del launch["developer_training_evidence"]
            elif change=="missing_evidence_pin":del launch["developer_training_evidence"]["input_sha256"][path]
            elif change=="changed_evidence_pin":launch["developer_training_evidence"]["input_sha256"][path]="f"*64
            elif change=="extra_evidence_pin":launch["developer_training_evidence"]["input_sha256"]["/unknown"]="f"*64
            elif change=="missing_launch_pin":del launch["input_sha256"][path]
            elif change=="changed_launch_pin":launch["input_sha256"][path]="f"*64
            else:launch["hardware_profile"]="rtx4090"
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.validate_developer_training_evidence(launch)

if __name__=="__main__":
    unittest.main()
