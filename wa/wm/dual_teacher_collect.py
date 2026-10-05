"""Independent start-state paired teacher collection; explicitly in-set, not heldout."""
import argparse, copy, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
import habitat, evt_bench, trained_agent
from evt_full_20260926.common import BENCH, SCENES, sha, scene
from lightnav_transport_20260927.agent import Agent as LightNav
from scripts.da3.teacher_recorder import TeacherRecorder
from wa.wm.oracle_teacher import OracleTeacher
from wa.wm.recovery_replay import dynamic_state, check_dynamic_state, rgb_hash
from wa.wm.dual_teacher_selection import EXPERIMENT, select_teacher
from wa.wm.semantic_scene import prepare_episode
from wa.wm.initial_bbox_repair import load_plan, repaired_detector

class BoundTeacher:
    save_video = False
    verbose_steps = False
    initialization_valid = True
    policy_failure_reason = None
    last_trajectory = None
    def __init__(self, teacher, allow_released_fallback=False):
        self.teacher = teacher
        self.environment = None
        self.fallback_count = 0
        self.allow_released_fallback = allow_released_fallback
        self.fallback_events = []
        self.action_step = 0
    def bind_environment(self, env):
        self.environment = env
        if hasattr(self.teacher, "bind_environment"):
            self.teacher.bind_environment(env)
    def reset(self, *args, **kwargs):
        self.teacher.reset(*args, **kwargs)
    def act(self, observations, detector, episode_id, instruction=None):
        action = np.asarray(self.teacher.act(observations, detector, episode_id, instruction), dtype=float)
        count=getattr(self.teacher,"episode_fallback_count",0)
        if count>self.fallback_count or getattr(self.teacher,"reply_error",None) is not None:
            if not self.allow_released_fallback:raise RuntimeError("TEACHER_TRANSPORT_FALLBACK")
            self.fallback_events.append(dict(step=self.action_step,action=action.tolist(),
                error=getattr(self.teacher,"reply_error",None)))
        self.fallback_count=count
        self.action_step+=1
        if action.shape != (3,) or not np.isfinite(action).all() or (abs(action)>1).any():
            raise ValueError("invalid teacher action")
        self.last_trajectory = getattr(self.teacher, "last_trajectory", None)
        return action.tolist()

class AdaptationRecorder(TeacherRecorder):
    def __init__(self, root, metadata, agent, repair=None, expected=None):
        assert metadata["experiment"] == EXPERIMENT
        super().__init__(root, metadata)
        self.agent, self.repair, self.expected = agent, repair, expected
        self.first = None
    def observe(self, obs, detector, robot, human, episode, step, frequency, world_time=None):
        if step == 0:
            self.first = dict(rgb=rgb_hash(obs["agent_1_articulated_agent_jaw_rgb"]),
                              state=dynamic_state(self.agent.environment))
            if self.expected is not None:
                if self.expected["rgb"] != self.first["rgb"]:
                    raise ValueError("paired initial RGB differs")
                check_dynamic_state(self.expected["state"], self.first["state"])
            if self.repair:
                detector = repaired_detector(detector, obs["agent_1_articulated_agent_jaw_rgb"], self.repair)
        super().observe(obs,detector,robot,human,episode,step,frequency,world_time)
        if step == 0 and self.repair:
            self.metadata.update(initial_bbox_rgb_xyxy=self.repair["bbox"],
                                 initial_bbox_status="VERIFIED_FROZEN_FIRST_RGB_REPAIR")
    def finish(self, result):
        super().finish(result)
        # No unsuccessful branch is eligible; preserve its raw records for diagnosis.
        path=self.root/"windows.json"
        windows=json.loads(path.read_text())
        (self.root/"raw_windows.json").write_text(json.dumps(windows))
        (self.root/"fallback_events.json").write_text(json.dumps(self.agent.fallback_events))
        if not result["success"] or result["collision"] or self.agent.fallback_events:
            path.write_text("[]")
        (self.root/"pair_start.json").write_text(json.dumps(self.first))
        (self.root/"admission.json").write_text(json.dumps(dict(
            experiment=EXPERIMENT, training_released=False,
            success=bool(result["success"]), pending="pair selection and SE2 label audit")))

def main():
    p=argparse.ArgumentParser()
    for name in ("manifest","output","repair-plan","repair-sha"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--development-one",action="store_true")
    p.add_argument("--development-key")
    p.add_argument("--development-three",action="store_true")
    p.add_argument("--teacher-url")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=8)
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--resume-plan")
    p.add_argument("--resume-sha")
    a=p.parse_args()
    resume=None
    if bool(a.resume_plan)!=bool(a.resume_sha):raise ValueError('resume plan/hash required together')
    if a.resume_plan:
        from wa.wm.dual_teacher_resume import load_resume
        resume=load_resume(a.resume_plan,a.resume_sha)
        if a.shards!=8 or a.development_one or a.development_three:raise ValueError('resume requires eight complete lanes')
    if a.development_key and not a.development_one:raise ValueError("key override only in developer one-pair check")
    if a.development_one or a.development_three:
        import os
        assert os.environ.get("WA_DEVELOPMENT")=="1" and not os.environ.get("MD_AK_JOB_ID")
    assert 0<=a.shard<a.shards
    m=json.loads(Path(a.manifest).read_text())
    if resume and sha(a.manifest)!=resume['manifest_sha256']:raise ValueError('resume data differs')
    planned={(r['task'],r['key']) for r in resume['lanes'][a.shard]} if resume else None
    repairs=load_plan(a.repair_plan,a.repair_sha)
    fixes={(r["task"],r["key"]):r for r in repairs["repairs"]}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    protocol=hashlib.sha256(("mp3d_semantic_ply_v1:"+a.repair_sha+":"+sha(a.manifest)).encode()).hexdigest()
    count=0
    for task in ("stt","dt","at"):
        spec=m["tasks"][task];assert sha(spec["path"])==spec["sha256"]
        cfg=habitat.get_config(str(BENCH/f"habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml"),[
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
            "habitat.environment.iterator_options.shuffle=false",
            "habitat.dataset.data_path="+spec["path"],"habitat.simulator.scene_dataset="+SCENES])
        from omegaconf import OmegaConf
        sensors=OmegaConf.to_container(cfg.habitat.simulator.agents.agent_1.sim_sensors,resolve=True)
        for field in ("height","width","position","orientation","hfov","sensor_subtype"):
            if sensors["jaw_rgb_sensor"][field] != sensors["jaw_panoptic_sensor"][field]:
                raise ValueError("RGB/semantic camera mismatch: "+field)
        ds=habitat.make_dataset(cfg.habitat.dataset.type,config=cfg.habitat.dataset)
        actual={scene(dict(scene_id=e.scene_id))+"/"+str(e.episode_id):e for e in ds.episodes}
        assert len(actual)==1405 and set(actual)=={e["key"] for e in spec["episodes"]}
        entries=[e for i,e in enumerate(spec["episodes"]) if i%a.shards==a.shard]
        if planned is not None:
            entries=[e for e in spec['episodes'] if (task,e['key']) in planned]
        for entry in entries:
            key=entry["key"];assert actual[key].info["instruction"]==entry["instruction"]
            if a.development_key and key!=a.development_key:continue
            if a.development_three and key!="pRbA3pwrgk9/38":continue
            if a.development_one and count>0:continue
            count+=1
            if a.audit_only:continue
            base=out/task/key;base.mkdir(parents=True)
            branches={};first=None
            for name in ("lightnav","oracle"):
                random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
                teacher=LightNav(str(base/"lightnav_client"),a.teacher_url) if name=="lightnav" else OracleTeacher()
                agent=BoundTeacher(teacher,allow_released_fallback=name=="lightnav")
                rec=AdaptationRecorder(base/name,dict(experiment=EXPERIMENT,partition="evaluation_adaptation",
                    task=task,key=key,teacher=name,camera_alignment_verified=True,
                    scene_id=actual[key].scene_id,episode_id=str(actual[key].episode_id),
                    source_dataset_sha256=spec["sha256"],protocol_sha256=protocol),
                    agent,fixes.get((task,key)),first)
                subset=copy.copy(ds);subset.episodes=[prepare_episode(actual[key],BENCH)]
                try:
                    trained_agent.evaluate_agent(copy.deepcopy(cfg),subset,str(base/(name+"_metrics")),
                                                 agent_factory=lambda _:agent,recorder=rec)
                finally:teacher.close()
                if first is None:first=rec.first
                # Tolerance-verified second branch refers to the exact first-branch state evidence.
                state_sha=hashlib.sha256(json.dumps(first["state"],sort_keys=True).encode()).hexdigest()
                result=json.loads((base/name/"result.json").read_text())
                branches[name]=dict(experiment=EXPERIMENT,teacher=name,complete=True,replay_verified=True,
                    transport_fallback=bool(agent.fallback_count),task=task,key=key,takeover_step=0,seed=7,
                    fallback_policy="released_lightnav_client_v1" if name=="lightnav" else None,
                    fallback_events=agent.fallback_events,
                    protocol_sha256=protocol,initial_rgb_sha256=first["rgb"],takeover_state_sha256=state_sha,
                    result=result,artifact_root=str(base/name))
            selection=select_teacher(branches["lightnav"],branches["oracle"])
            selection["branches"]=branches
            (base/"selection.json").write_text(json.dumps(selection,indent=2))
            with (out/"selections.jsonl").open("a") as f:f.write(json.dumps(selection)+"\n")
            print("PAIR_COMPLETE",task,key,selection["selected_teacher"],flush=True)
    if planned is not None and count!=len(planned):raise ValueError('resume lane incomplete')
    (out/("AUDIT_PASS.json" if a.audit_only else "COMPLETE.json")).write_text(
        json.dumps(dict(pairs=count,experiment=EXPERIMENT,training_released=False)))
if __name__=="__main__":main()
