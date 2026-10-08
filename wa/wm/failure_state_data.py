"""Independent, explicitly SHA-pinned admission for failure-state training data.

No flag on disk is upgraded to training_released. Only original successful
teacher suffixes supply action labels; student prefix may supply causal history.
Unconsumed historical weights retain the prior collection audit's SHA chain,
rather than being rehashed by every training rank.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from wa.wm.robot_data import RobotWorldData, CONTRACT, transition_records
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.wm.failure_state_protocol import EXPERIMENT as V1, PROTOCOL_SHA as P1
from wa.wm.failure_state_protocol_v2 import EXPERIMENT as V2, PROTOCOL_SHA as P2
from wa.wm.failure_state_boundary_policy import CONTINUATION_POLICY

EXPECTED = dict(completed_searches=126, accepted_original_episodes=96,
                candidate_windows=7396, valid_windows=6864, excluded_windows=532,
                episodes_with_valid_windows=90)
SCHEMA = "failure_state_cache_v1"
EXPERIMENT = "evaluation_adaptation_failure_state_cache_v1"
SUFFIXES = ("pose.npy", "history.npy", "episode.npy", "episodes.json")
CACHE_FILES = {f"{part}_{suffix}" for part in ("train", "heldout") for suffix in SUFFIXES}
INDEX_FILES = {"index/train_valid.npy", "index/train_episodes_audit.json",
               "index/heldout_valid.npy", "index/heldout_episodes_audit.json", "index/audit.json"}
FILES = CACHE_FILES | INDEX_FILES | {"source_image_hashes.json", "source_files.json", "complete.json"}
HELDOUT = {f"heldout_{suffix}" for suffix in SUFFIXES} | {
    "index/heldout_valid.npy", "index/heldout_episodes_audit.json"}
SOURCE_JOBS = {(61833,73055):(V1,P1), (61836,73058):(V2,P2), (61844,73066):(V2,P2)}
RAW_FILES = {"metadata.json", "observations.json", "actions.json", "windows.json",
             "result.json", "branch.json", "admission.json", "complete.json",
             "replay.json", "first_start.json", "pair_start.json", "takeover.json",
             "fallback_events.json", "raw_windows.json"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b, name):
    require(json.dumps(a, sort_keys=True, allow_nan=False) ==
            json.dumps(b, sort_keys=True, allow_nan=False), name)


def strict_json(blob):
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def invalid(x):
        raise ValueError("nonfinite JSON: " + x)
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=invalid)


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
            "explicit lowercase SHA256 required")
    return value


def directory(value):
    require(type(value) in (str, type(Path())) or isinstance(value, Path), "directory path required")
    p = Path(value)
    require(p.is_absolute() and p.is_dir() and not p.is_symlink() and p.resolve() == p,
            "absolute nonsymlink directory required")
    return p


def success(result):
    require(isinstance(result, dict), "terminal result required")
    for name in ("success","collision","policy_init_valid"):
        v = result.get(name)
        require(type(v) in (bool,int,float) and v in (0,1), "binary result required")
    require(bool(result["success"]) and not result["collision"] and result["policy_init_valid"],
            "only successful collision-free initialized original branches")

START_EVIDENCE_FIELDS = ("step", "observation_index", "timestamp_s",
                        "simulator_world_time_s", "rgb_sha256",
                        "dynamic_state", "dynamic_state_sha256")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_start_evidence(doc, meta, observations, actions, entry, artifact_names):
    """Recheck actual producer start/pair/takeover evidence after SHA admission.

    The separately pinned collection release already compares to original61609,
    the student prefix and repeat. This checks internal first0/k evidence and
    executed-action proof; it does not recreate hidden simulator/RNG state or
    confuse the recorded raw-sensor digest with the encoded PNG file digest.
    """
    require(not {"first_start_pair.json", "takeover_pair.json", "teacher_error.json"}
            & set(artifact_names), "non-producer/teacher-error evidence in original winner")
    if meta.get("initial_bbox_status") == "VERIFIED_CONFIG_AND_SEMANTIC":
        require("initial_panoptic.npy" in artifact_names, "missing pinned initial semantic evidence")
    first, pair, takeover, replay = (doc(name+".json") for name in
                                      ("first_start", "pair_start", "takeover", "replay"))
    admission, branch, result = (doc(name+".json") for name in ("admission", "branch", "result"))
    same(doc("fallback_events.json"), [], "normal winner has fallback/error events")
    require(isinstance(doc("raw_windows.json"), list), "raw windows evidence required")
    same(admission.get("issues"), [], "winner recorder issues")
    require(isinstance(first, dict) and isinstance(pair, dict) and isinstance(takeover, dict),
            "real first/pair/takeover observations required")
    same(pair, first, "pair_start must equal episode-zero first_start")
    same(sorted(first), sorted(set(START_EVIDENCE_FIELDS) |
                              {"initial_rgb_sha256", "initial_bbox_sensor_xyxy_original"}),
         "first_start producer fields")
    same(sorted(takeover), sorted(set(START_EVIDENCE_FIELDS) |
         {"initial_rgb_sha256", "initial_bbox_sensor_xyxy_original",
          "expected_rgb_sha256", "expected_dynamic_state_sha256"}), "takeover producer fields")
    n = len(observations); k = entry.get("takeover_step")
    require(type(k) is int and 0 <= k < n and isinstance(replay, list)
            and len(replay) == len(actions) == n, "complete replay/action/observation counts")
    require(type(result.get("total_step")) is int and result["total_step"] == n,
            "terminal action count differs")
    prefix_sha = digest(meta.get("prefix_sha256"))
    original_box = meta.get("initial_bbox_sensor_xyxy_original")
    box = np.asarray(original_box)
    require(box.shape == (4,) and box.dtype.kind in "iuf" and np.isfinite(box).all(),
            "original episode-zero sensor bbox")
    same(meta.get("initial_bbox_sensor_xyxy"), original_box, "original sensor bbox replaced")
    for i, (r, observation, action) in enumerate(zip(replay, observations, actions)):
        require(isinstance(r, dict) and type(r.get("step")) is int and r["step"] == i
                and type(r.get("observation_index")) is int and r["observation_index"] == i,
                "consecutive replay observation indices")
        for field in ("timestamp_s", "simulator_world_time_s"):
            value = r.get(field)
            require(type(value) in (int,float) and np.isfinite(value), "finite replay time")
            same(value, observation.get(field), "replay/raw observation time")
        state = r.get("dynamic_state")
        require(isinstance(state,dict) and isinstance(state.get("agents"),list)
                and bool(state["agents"]), "dynamic state structure")
        same(state.get("timestamp"), r["simulator_world_time_s"], "dynamic world time")
        same(digest(r.get("dynamic_state_sha256")), canonical_digest(state), "dynamic state digest")
        digest(r.get("rgb_sha256"))
        same(r.get("action"), action.get("normalized_action"), "recorded replay action differs")
        same(r.get("normalized_action"), action.get("normalized_action"), "replay normalized action differs")
        same(r.get("owner"), action.get("owner"), "replay action owner")
        same(r.get("owner"), "student" if i<k else "teacher", "replay takeover ownership")
        require(r.get("action_timing") == "before_env_step" and r.get("post_state_recorded") is False
                and "post_state" not in r, "fabricated replay poststate/timing")
    for evidence, index in ((first,0),(takeover,k)):
        for field in START_EVIDENCE_FIELDS:
            same(evidence.get(field), replay[index].get(field), "start/takeover replay "+field)
        same(evidence.get("initial_bbox_sensor_xyxy_original"), original_box,
             "start/takeover changed episode-zero bbox")
        same(evidence.get("initial_rgb_sha256"), replay[0]["rgb_sha256"],
             "start/takeover changed episode-zero raw RGB")
    same(takeover.get("expected_rgb_sha256"), takeover["rgb_sha256"],
         "takeover expected raw RGB differs")
    expected_takeover = digest(takeover.get("expected_dynamic_state_sha256"))
    actual_takeover = takeover["dynamic_state_sha256"]
    # Expected/actual state SHA need not equal when replay tolerance is <=1e-6;
    # the pinned full branch audit proves that comparison against student state.
    links = dict(prefix_sha256=prefix_sha,initial_rgb_sha256=first["rgb_sha256"],
                 takeover_state_sha256=expected_takeover,
                 actual_takeover_state_sha256=actual_takeover)
    for record in (admission,branch):
        for field,value in links.items():
            same(record.get(field),value,"start pair identity "+field)
    same(admission.get("takeover_observation_index"),k,"admission takeover observation")
    proof = admission.get("agent_validation")
    require(isinstance(proof,dict),"recorded replay wrapper proof required")
    for field in ("replay_wrapper","prefix_hash_matches","environment_bound",
                  "takeover_matches","replay_verified"):
        require(proof.get(field) is True,"missing replay proof "+field)
    for field,value in (("agent_step_before_reset",n),("verified_prefix_frames",k+1),
                        ("required_prefix_frames",k+1)):
        require(type(proof.get(field)) is int and proof[field] == value,"replay proof "+field)
    require(proof.get("hidden_rng_contact_state_proven") is False,"unsupported hidden-state proof")
    same(proof.get("verification_source"),
         "successful ReplayThenTeacher.act calls before final reset","proof source")
    same(proof.get("proof_scope"),
         "t=0..k raw RGB plus world time and articulated transforms/joints, atol=1e-6","proof scope")
    same(admission.get("owned_suffix"),dict(action_count=n-k,start_observation_index=k,
         end_observation_index_inclusive=n-1,start_step=k,end_step_inclusive=n-1,
         contiguous=True,poststate_claim="none; only next preaction frames are observed"),
         "real teacher suffix ownership")
    return dict(frames=n,takeover_step=k,verified_prefix_frames=k+1,
                initial_rgb_sha256=first["rgb_sha256"],actual_takeover_state_sha256=actual_takeover)


class Reader:
    def __init__(self):
        self.hashes, self.states = {}, {}

    def read(self, value, expected, *, retain=False):
        p = Path(value); digest(expected)
        require(p.is_absolute() and p.is_file() and not p.is_symlink() and p.resolve() == p,
                "absolute nonsymlink file required: " + str(p))
        s = p.stat(); state = (s.st_size,s.st_mtime_ns,s.st_ino)
        name = str(p)
        require(name not in self.states or self.states[name] == state, "source changed during admission")
        h, chunks = hashlib.sha256(), []
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                h.update(chunk)
                if retain: chunks.append(chunk)
        after = p.stat()
        require(not p.is_symlink() and p.resolve() == p and
                (after.st_size,after.st_mtime_ns,after.st_ino) == state, "source changed while hashing")
        require(h.hexdigest() == expected, "source SHA mismatch: " + str(p))
        self.hashes[name],self.states[name] = expected,state
        return b"".join(chunks) if retain else expected

    def doc(self, path, expected):
        return strict_json(self.read(path,expected,retain=True))

    def unchanged(self, paths=None):
        for name in self.states if paths is None else map(str,paths):
            p=Path(name);s=p.stat()
            require(not p.is_symlink() and p.resolve()==p and
                    (s.st_size,s.st_mtime_ns,s.st_ino)==self.states[name],
                    "admitted source changed before use: "+name)


class FailureStateData(RobotWorldData):
    """Production-only fixed126 cache. Admission SHA is a required caller pin."""
    def __init__(self, cache, *, expected_admission_sha256):
        self.cache = directory(cache)
        self._reader = Reader()
        admission_sha = digest(expected_admission_sha256)
        admission = self._reader.doc(self.cache/"admission.json",admission_sha)
        same(admission.get("schema"),SCHEMA+"_admission_candidate","admission schema")
        same(admission.get("status"),"CACHE_CONVERTED_NOT_TRAINING_RELEASED","admission status")
        for k in ("expected_admission_sha_required_by_loader","independent_audit_required"):
            require(admission.get(k) is True,"admission gate: "+k)
        require(admission.get("training_released") is False,"converter must not release training")
        same(admission.get("summary"),EXPECTED,"fixed admission counts")
        files=admission.get("files")
        require(isinstance(files,dict) and set(files)==FILES,"exact sixteen-file admission graph")
        original=admission.get("original_heldout",{})
        base,index=directory(original["cache"]),directory(original["index"])
        self._links={}
        for name,h in files.items():
            p=self.cache/name
            if name in HELDOUT:
                target=(index/p.name) if name.startswith("index/") else base/p.name
                require(p.is_symlink() and p.resolve()==target and target.resolve()==target,
                        "heldout must link to exact original file")
                self._links[str(p)]=str(target)
                p=target
            self._reader.read(p,h)
        complete=self._doc("complete.json",files)
        audit=self._doc("index/audit.json",files)
        same(complete.get("schema"),SCHEMA,"cache schema")
        for d in (complete,audit):
            same(d.get("experiment"),EXPERIMENT,"cache experiment")
            same(d.get("contract"),CONTRACT,"robot contract")
            require(d.get("training_released") is False,"cache prematurely released")
        same(audit.get("schema"),SCHEMA+"_index","index schema")
        same(complete.get("summary"),EXPECTED,"cache counts")
        same(complete.get("files"),{k:files[k] for k in CACHE_FILES},"eight cache SHA graph")
        same(audit.get("cache_complete_sha256"),files["complete.json"],"index cache binding")
        same(audit.get("train_episode_audit_sha256"),files["index/train_episodes_audit.json"],
             "train episode audit binding")
        same(audit.get("heldout_episode_audit_sha256"),files["index/heldout_episodes_audit.json"],
             "heldout episode audit binding")
        for field in ("original_frame_sequences_preserved","future_labels_teacher_owned",
                      "independent_cache_admission_required"):
            require(complete.get(field) is True,"cache declaration: "+field)
        for field in ("template_index","student_prefix_actions_used_as_future_labels","repeat_windows_counted"):
            require(type(complete.get(field)) is int and complete[field]==0,"cache zero declaration: "+field)
        same(complete.get("source_image_inventory_sha256"),files["source_image_hashes.json"],"image inventory pin")
        same(complete.get("source_file_inventory_sha256"),files["source_files.json"],"source inventory pin")
        same(admission.get("source_files_sha256"),files["source_files.json"],"admission source inventory")
        release_ref=admission.get("collection_release",{})
        same(release_ref,dict(path=complete.get("source_release"),sha256=complete.get("source_sha256")),
             "release cross-binding")
        release=self._reader.doc(release_ref["path"],release_ref["sha256"])
        same(release.get("schema"),"failure_state_collection_release_v1","collection release schema")
        for field in ("collection_validated","evaluation_adaptation","no_success_rate","cache_conversion_required"):
            require(release.get(field) is True,"collection admission: "+field)
        for field in ("training_released","untouched_test","score_backfill_allowed"):
            require(release.get(field) is False,"unsafe collection declaration")
        same(release.get("expected"),126,"full collection expected")
        same(release.get("completed_searches"),126,"full collection complete")
        same(release.get("summary"),{k:v for k,v in EXPECTED.items() if k!="completed_searches"},
             "release summary")
        source_files=self._doc("source_files.json",files)
        require(isinstance(source_files,dict) and source_files,"source inventory")
        for path,h in source_files.items():
            require(type(path) is str and Path(path).is_absolute(),"absolute source inventory key")
            digest(h)
        require(isinstance(release.get("source_files"),dict) and release["source_files"],"release source graph")
        for path,h in release["source_files"].items():
            same(source_files.get(path),h,"release source inventory mapping")
        same(source_files.get(release_ref["path"]),release_ref["sha256"],"release is a source pin")
        same(complete.get("code_sha256"),admission.get("code_sha256"),"converter code cross-binding")
        require(isinstance(complete.get("code_sha256"),dict) and complete["code_sha256"],"converter code provenance")
        for path,h in complete["code_sha256"].items():
            same(source_files.get(path),h,"code inventory binding")
            self._reader.read(path,h)
        # Only the unchanged heldout is linked, never a new recovery heldout.
        original_pins={str(base/"complete.json"):original["cache_complete_sha256"],
                       str(index/"audit.json"):original["index_audit_sha256"]}
        original_pins.update({self._links[str(self.cache/name)]:files[name] for name in HELDOUT})
        for path,h in original_pins.items():
            same(source_files.get(path),h,"original heldout source inventory binding")
        base_complete=self._reader.doc(base/"complete.json",original["cache_complete_sha256"])
        base_audit=self._reader.doc(index/"audit.json",original["index_audit_sha256"])
        same(base_audit.get("contract"),CONTRACT,"original index contract")
        same(base_audit.get("cache_complete_sha256"),original["cache_complete_sha256"],"original cache/index pin")
        for suffix in SUFFIXES:
            same(base_complete["files"]["heldout_"+suffix],files["heldout_"+suffix],"original heldout file")
        same(audit["splits"]["heldout"],base_audit["splits"]["heldout"],"original heldout split")
        same(files["index/heldout_valid.npy"],base_audit["splits"]["heldout"]["index_sha256"],"heldout valid pin")
        train_audit=self._doc("index/train_episodes_audit.json",files)
        require(train_audit.get("errors")==[],"episode audit errors")
        entries=self._doc("train_episodes.json",files)
        demos=release.get("teacher_demonstrations")
        stats=train_audit.get("stats")
        require(isinstance(demos,list) and isinstance(entries,list) and isinstance(stats,list)
                and len(demos)==len(entries)==len(stats)==96,"exact original winner count")
        images=self._doc("source_image_hashes.json",files)
        require(isinstance(images,dict) and set(images)=={d["branch"] for d in demos},
                "exact image branch inventory")
        self._admissions={}
        self._episode_paths={}
        seen_keys,seen_roots,seen_repeats=set(),set(),set()
        pose=np.load(self.cache/"train_pose.npy",mmap_mode="r",allow_pickle=False)
        history=np.load(self.cache/"train_history.npy",mmap_mode="r",allow_pickle=False)
        episode=np.load(self.cache/"train_episode.npy",mmap_mode="r",allow_pickle=False)
        valid=np.load(self.cache/"index/train_valid.npy",allow_pickle=False)
        require(pose.dtype==np.float32 and pose.shape==(7396,7,4) and np.isfinite(pose).all(),
                "pose cache shape/type/finite")
        require(history.dtype==np.int32 and history.shape==(7396,4)
                and episode.dtype==np.int32 and episode.shape==(7396,)
                and valid.dtype==np.int64 and valid.shape==(6864,), "index array schema")
        require((np.diff(valid)>0).all() and (valid>=0).all() and (valid<7396).all(),
                "valid rows unique ordered in range")
        offset=0; derived_valid=[];nonempty=0;heldscenes=set()
        for e in self._doc("heldout_episodes.json",files):
            heldscenes.add(Path(e["scene"]).name.split(".")[0])
        overlap=set()
        for ep,(d,entry,stat) in enumerate(zip(demos,entries,stats)):
            root=directory(d["branch"]);repeat=directory(d["repeat_branch"])
            require(type(d.get("key")) is str and not Path(d["key"]).is_absolute()
                    and len(Path(d["key"]).parts)==2 and ".." not in Path(d["key"]).parts,
                    "episode key")
            require(d.get("task")=="stt" and d.get("teacher") in ("oracle","lightnav")
                    and type(d.get("takeover_step")) is int and d["takeover_step"]>=0,"demo identity")
            key=(d["task"],d["key"])
            require(key not in seen_keys and str(root) not in seen_roots and str(repeat) not in seen_repeats,
                    "duplicate winner identity")
            seen_keys.add(key);seen_roots.add(str(root));seen_repeats.add(str(repeat))
            require(root.name==f'{d["teacher"]}_{d["takeover_step"]:04d}'
                    and repeat==root.with_name(root.name+"_repeat")
                    and str(root.parent).endswith("/stt/"+d["key"]),"original branch path identity")
            pair=(d.get("source_job_id"),d.get("source_task_id"))
            require(all(type(x) is int for x in pair) and pair in SOURCE_JOBS,"source job identity")
            same((d.get("source_experiment"),d.get("source_protocol_sha256")),SOURCE_JOBS[pair],"source protocol")
            if pair==(61844,73066):
                same(d.get("source_boundary_policy"),CONTINUATION_POLICY,"v3 boundary policy")
            else:
                require("source_boundary_policy" not in d,"old source policy relabeled")
            hashes=d.get("hashes")
            require(isinstance(hashes,dict) and RAW_FILES<=set(hashes),"complete original source evidence")
            paths=[]
            for name,h in hashes.items():
                rel=Path(name)
                require(type(name) is str and not rel.is_absolute() and ".." not in rel.parts
                        and "." not in rel.parts,"source relative path")
                path=root/rel
                same(source_files.get(str(path)),h,"consumed source inventory binding")
                same(release["source_files"].get(str(path)),h,"consumed release binding")
                self._reader.read(path,h);paths.append(path)
            def doc(name):return self._reader.doc(root/name,hashes[name])
            meta,obs,acts,wins=(doc(n) for n in ("metadata.json","observations.json","actions.json","windows.json"))
            for record in (meta,doc("branch.json"),doc("admission.json")):
                for field,value in dict(experiment=d["source_experiment"],protocol_sha256=d["source_protocol_sha256"],
                        task=d["task"],key=d["key"],teacher=d["teacher"],takeover_step=d["takeover_step"]).items():
                    same(record.get(field),value,"raw identity "+field)
                require(record.get("verification_only") is False,"repeat cannot teach")
            self._admissions[str(root)]=d
            self.validate_identity(meta,root)
            result=doc("result.json");success(result)
            for record in (doc("branch.json"),doc("admission.json")):
                require(record.get("complete") is True and record.get("replay_verified") is True
                        and record.get("transport_fallback") is False,"incomplete/fallback teacher")
                same(record.get("result"),result,"terminal disagreement")
            raw_admission=doc("admission.json")
            require(raw_admission.get("training_eligible") is False
                    and raw_admission.get("training_released") is False,"raw admission prematurely released")
            same(doc("branch.json").get("artifact_root"),str(root),"raw original root")
            raw_complete=doc("complete.json")
            require(raw_complete.get("complete") is True
                    and raw_complete.get("status")=="FAILURE_STATE_RAW_COLLECTION_COMPLETE"
                    and raw_complete.get("training_eligible") is False
                    and raw_complete.get("training_released") is False,"raw completion")
            require(isinstance(obs,list) and len(obs)>1 and raw_complete.get("frames")==len(obs),"observation count")
            validate_start_evidence(doc,meta,obs,acts,d,hashes)
            frames=[o["frame"] for o in obs]
            same(frames,[f"rgb_{i:04d}.png" for i in range(len(obs))],"full episode-zero frame sequence")
            same(images[str(root)],{f:hashes[f] for f in frames},"exact image inventory")
            target=np.asarray([o.get("target_position_world_label_only") for o in obs])
            require(target.shape==(len(obs),3) and target.dtype.kind in "iuf"
                    and np.isfinite(target).all(),"current ideal simulated UWB geometry")
            numeric=audit_numeric_windows(obs,acts,wins,d["takeover_step"],per_window=False)
            nd=numeric["derived"];count=len(wins)
            same(nd["ownership_audit"]["teacher"],d["teacher"],"executed teacher identity")
            same([w["current_index"] for w in wins],d.get("window_indices"),"exact candidate indices")
            same(numeric["valid_window_indices"],d.get("valid_window_indices"),"exact valid indices")
            same(numeric["excluded"],d.get("numeric_exclusions"),"exact exclusions")
            same(numeric["summary"]["candidate_windows"],d.get("candidate_windows"),"candidate count")
            same(numeric["summary"]["valid_windows"],d.get("expected_valid_count"),"valid count")
            same(numeric["summary"]["excluded_windows"],d.get("expected_excluded_count"),"excluded count")
            sl=slice(offset,offset+count)
            require(np.array_equal(pose[sl],nd["pose"]) and np.array_equal(history[sl],nd["history"])
                    and np.array_equal(episode[sl],np.full(count,ep,dtype=np.int32)),
                    "cache labels/history/episode differ from raw numeric derivation")
            rows=(offset+np.flatnonzero(numeric["effective_mask"])).tolist()
            expected_entry=dict(root=str(root),frames=frames,task=d["task"],episode_uid="stt:"+d["key"],
                scene=meta["scene_id"],takeover_step=d["takeover_step"],category="failure_state_evaluation_adaptation",
                teacher=d["teacher"],template_index=0,source_experiment=d["source_experiment"],
                source_protocol_sha256=d["source_protocol_sha256"],source_job_id=pair[0],source_task_id=pair[1],
                source_repeat_branch=str(repeat),repeat_frames_referenced=False,expected_valid_count=len(rows))
            if pair==(61844,73066):expected_entry["source_boundary_policy"]=CONTINUATION_POLICY
            same(entry,expected_entry,"exact cache episode identity/frames")
            commands,bad,transition=transition_records(root,obs)
            require(np.array_equal(commands,nd["commands"]) and np.array_equal(bad,nd["transition_bad"]),
                    "unchanged RobotWorldData transition contract")
            expected_stat={**transition,**numeric["summary"],"episode":ep,"rows":count,"retained":len(rows),
                "candidate_row_start":offset,"candidate_current_indices":d["window_indices"],
                "valid_current_indices":numeric["valid_window_indices"],"valid_cache_rows":rows,
                "numeric_exclusions":numeric["excluded"],
                "source_sha256":{n:hashes[n] for n in ("metadata.json","observations.json","actions.json")}}
            same(stat,expected_stat,"exact per-episode index audit")
            derived_valid.extend(rows);offset+=count;nonempty+=bool(rows)
            self._episode_paths[ep]=paths
            if Path(meta["scene_id"]).name.split(".")[0] in heldscenes:
                overlap.add(Path(meta["scene_id"]).name.split(".")[0])
        require(not seen_roots&seen_repeats,"repeat used as an original")
        require(offset==7396 and len(derived_valid)==6864 and nonempty==90,"derived fixed counts")
        require(np.array_equal(valid,np.asarray(derived_valid,dtype=np.int64)),"exact runtime valid rows")
        same(complete.get("known_heldout_scene_overlap"),sorted(overlap),"declared scene overlap")
        same(audit["splits"]["train"],dict(original=7396,retained=6864,excluded=532,
            index_sha256=files["index/train_valid.npy"],episode_errors=[]),"train split contract")
        self._reader.unchanged()
        self.admission_sha256=admission_sha
        self.admission_report=dict(status="LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT",
            summary=dict(EXPECTED),admission_sha256=admission_sha,release_sha256=release_ref["sha256"],
            actual_consumed_sources_hashed=True,consumed_source_files=len(self._reader.hashes),
            unconsumed_historical_weights_rehashed=False,
            input_scope="RGB/episode-zero BBox/current ideal zero-delay simulated polar UWB; no text",
            training_released_on_disk=False)
        super().__init__(self.cache,"train",self.cache/"index")
        raw_paths={str(p) for paths in self._episode_paths.values() for p in paths}
        self._cache_guard=[p for p in self._reader.states if p not in raw_paths]

    def _doc(self,name,files):
        path=self.cache/name
        return self._reader.doc(self._links.get(str(path),str(path)),files[name])

    def validate_identity(self,meta,root):
        d=self._admissions.get(str(root))
        require(d is not None,"unadmitted raw episode")
        require(meta.get("partition")=="evaluation_adaptation" and meta.get("training_eligible") is False
                and meta.get("camera_alignment_verified") is True,"unaligned/unmarked raw data")
        require(meta.get("initial_bbox_status") in ("VERIFIED_CONFIG_AND_SEMANTIC",
                "VERIFIED_FROZEN_FIRST_RGB_REPAIR"),"unverified episode-zero BBox")
        require(meta.get("timebase_version")=="v3_actual_world_time_interpolated","timebase")
        shape=np.asarray(meta.get("rgb_shape"));box=np.asarray(meta.get("initial_bbox_rgb_xyxy"))
        require(shape.shape==(3,) and shape.dtype.kind in "iu" and (shape>0).all()
                and shape[2]==3 and box.shape==(4,) and box.dtype.kind in "iuf" and np.isfinite(box).all()
                and 0<=box[0]<box[2]<=shape[1] and 0<=box[1]<box[3]<=shape[0],"template bbox/shape")
        if d["source_job_id"]==61844:
            same(meta.get("teacher_boundary_policy"),CONTINUATION_POLICY,"raw v3 boundary")
        else:
            require("teacher_boundary_policy" not in meta,"old metadata policy changed")

    def __getitem__(self,index):
        require(type(index) is int or isinstance(index,np.integer) and not isinstance(index,np.bool_),
                "integer dataset position required")
        if not 0<=index<len(self):raise IndexError(index)
        for link,target in self._links.items():
            require(Path(link).is_symlink() and Path(link).resolve()==Path(target),"heldout link changed")
        self._reader.unchanged(self._cache_guard)
        ep=int(self.episode[int(self.rows[index])])
        self._reader.unchanged(self._episode_paths[ep])
        item=super().__getitem__(index)
        require(set(item)=={"rgb","template","polar","times","pose","geometry","wm_rgb","commands","proprio","future"},
                "unexpected model input key")
        return item
