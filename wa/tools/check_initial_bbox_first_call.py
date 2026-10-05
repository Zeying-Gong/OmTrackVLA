"""Bounded developer check of actual evaluate_agent first-call path; no action or model inference."""
import copy,json,os,random,sys
from pathlib import Path
import numpy as np,torch,habitat,evt_bench,trained_agent
from wa.wm.initial_bbox_repair import load_plan,repaired_detector
from wa.wm.semantic_scene import prepare_episode,validate_runtime
from evt_full_20260926.common import BENCH,SCENES,scene,sha
class FirstCallValidated(Exception):pass
class FirstCallAgent(trained_agent.DA3EVTAgent):
    def __init__(self,repair):self.repair=repair
    def reset(self):self.sim_step=0
    def bind_environment(self,env):self.env=env
    def act(self,observations,detector,episode_id,instruction=None,timestamp_s=None):
        assert str(episode_id)==self.repair["key"].split("/")[-1]
        validate_runtime(self.env.sim,self.env.current_episode)
        d=repaired_detector(detector,observations["agent_1_articulated_agent_jaw_rgb"],self.repair)
        assert list(self._bbox(d))==self.repair["bbox"]
        raise FirstCallValidated()
def main():
    plan=load_plan(sys.argv[1],sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(exist_ok=False)
    root=Path("/data/nas_ray/home/zeying.gong/algorithm/repos")
    m=json.loads((root/"WLA-EVT-20260925/evt_full_20260926/manifest.json").read_text())
    done=[]
    for task in ("stt","dt","at"):
        spec=m["tasks"][task];assert sha(spec["path"])==spec["sha256"]
        config=habitat.get_config(str(BENCH/f"habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml"),[
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0","habitat.environment.iterator_options.shuffle=false",
            "habitat.dataset.data_path="+spec["path"],"habitat.simulator.scene_dataset="+SCENES])
        ds=habitat.make_dataset(id_dataset=config.habitat.dataset.type,config=config.habitat.dataset)
        actual={scene(dict(scene_id=e.scene_id))+"/"+str(e.episode_id):e for e in ds.episodes}
        for r in (r for r in plan["repairs"] if r["task"]==task):
            ep=prepare_episode(actual[r["key"]],BENCH);subset=copy.copy(ds);subset.episodes=[ep]
            random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            agent=FirstCallAgent(r)
            try:trained_agent.evaluate_agent(copy.deepcopy(config),subset,str(out/task),agent_factory=lambda _:agent)
            except FirstCallValidated:
                done.append({"task":task,"key":r["key"],"firstcall_pass":True})
                print("FIRST_CALL_PASS",task,r["key"],flush=True)
            else:raise AssertionError("unexpected closed-loop execution")
    assert len(done)==7
    (out/"PASS.json").write_text(json.dumps(done,indent=2))
if __name__=="__main__":main()
