"""Synthetic exact-consumption dedup; no GPU and no real scientific output."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from wa.tools import audit_failure_state_dedup as mod
from wa.tools import build_failure_state_cache as builder
from wa.tests.test_build_failure_state_cache import fixture, save


class DedupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.f=fixture(self.root)
        with patch.object(builder,"NAS",self.root),patch.object(builder,"EXPECTED",self.f["expected"]):
            builder.convert(self.f["release"],builder.file_sha(self.f["release"]),self.f["base"],
                builder.file_sha(self.f["base"]/"complete.json"),self.f["index"],
                builder.file_sha(self.f["index"]/"audit.json"),self.f["out"])
        self.old=self.f["out"]
        self.release=json.loads(self.f["release"].read_text())
        newfiles={}
        for entry in self.release["teacher_demonstrations"]:
            old=Path(entry["branch"]);new=self.root/"new"/entry["key"]/old.name
            new.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(old,new)
            repeat=new.parent/(new.name+"_repeat");repeat.mkdir()
            entry["branch"]=str(new);entry["repeat_branch"]=str(repeat)
            newfiles.update({str(new/n):h for n,h in entry["hashes"].items()})
        self.release["source_files"]=newfiles
        self.path=self.root/"new-release.json";self.sync()
        self.addCleanup(patch.stopall)
        patch.object(mod,"EXPECTED",dict(episodes=2,candidate_windows=7,valid_windows=3)).start()
        patch.object(mod,"OLD_COMPLETE_SHA",builder.file_sha(self.old/"complete.json")).start()
        self.anchor=self.root/"old-training-audit.json"
        save(self.anchor,dict(status="TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY",
            source_hashes={str(self.old/n):builder.file_sha(self.old/n) for n in
                          ("complete.json","index/audit.json","index/train_episodes_audit.json")}))
        patch.object(mod,"OLD_TRAINING_AUDIT",self.anchor).start()
        patch.object(mod,"OLD_TRAINING_AUDIT_SHA",builder.file_sha(self.anchor)).start()

    def sync(self):save(self.path,self.release)

    def repin(self,name,eid=0):
        e=self.release["teacher_demonstrations"][eid];p=Path(e["branch"])/name
        h=builder.file_sha(p);e["hashes"][name]=h;self.release["source_files"][str(p)]=h;self.sync()

    def change(self,name,fn,eid=0):
        e=self.release["teacher_demonstrations"][eid];p=Path(e["branch"])/name
        d=json.loads(p.read_text());fn(d);save(p,d);self.repin(name,eid)

    def audit(self):
        return mod.audit(self.path,builder.file_sha(self.path),self.old,
                         builder.file_sha(self.old/"index/audit.json"),progress=False)

    def test_all_consumed_fields_and_mapping(self):
        r=self.audit()
        self.assertEqual(r["summary"]["exact_duplicate_count"],3)
        self.assertEqual(r["duplicate_new_cache_rows"],[0,1,2])
        self.assertEqual([w["old_matching_rows"] for w in r["windows"]],[[0],[1],[2]])
        self.assertEqual(set(r["windows"][0]["nonimage_components"]),set(mod.NONIMAGE))
        self.assertEqual(set(r["windows"][0]["consumed_image_components"]),set(mod.IMAGES))
        self.assertFalse(r["sampling_changed"]);self.assertFalse(r["training_released"])
        self.assertNotIn("admission.json",r)
        helper=str(Path(builder.__file__).resolve())
        self.assertEqual(r["code_sha256"][helper],builder.file_sha(helper))
        self.assertEqual(r["source_files"][helper],builder.file_sha(helper))

    def test_teacher_identity_not_a_filter(self):
        e=self.release["teacher_demonstrations"][0];e["teacher"]="lightnav"
        self.change("metadata.json",lambda d:d.update(teacher="lightnav"))
        self.change("actions.json",lambda rows:[a.update(teacher="lightnav") for a in rows if a["owner"]=="teacher"])
        r=self.audit();self.assertEqual(r["summary"]["exact_duplicate_count"],3)
        self.assertEqual(r["windows"][0]["old_matches"][0]["teacher"],"oracle")

    def test_same_future_different_student_prefix_image_is_not_duplicate(self):
        e=self.release["teacher_demonstrations"][0];p=Path(e["branch"])/"rgb_0003.png"
        Image.new("RGB",(8,8),(255,200,100)).save(p);self.repin(p.name)
        r=self.audit();self.assertEqual(r["summary"]["exact_duplicate_count"],2)
        first=r["windows"][0]
        self.assertEqual(first["status"],"CONSUMED_IMAGE_DIFFERENCE")
        self.assertEqual(first["image_mismatch_fields"],{"0":["wm_rgb"]})

    def test_polar_difference_prefilters_without_decoding_that_row(self):
        self.change("observations.json",lambda rows:rows[6].update(target_position_world_label_only=[3.,0.,0.]))
        r=self.audit();self.assertEqual(r["summary"]["exact_duplicate_count"],2)
        self.assertEqual(r["windows"][0]["status"],"NO_NONIMAGE_MATCH")
        self.assertIsNone(r["windows"][0]["consumed_image_components"])

    def test_all_numeric_different_no_images_opened(self):
        self.change("observations.json",lambda rows:[o.update(target_position_world_label_only=[3.,0.,0.]) for o in rows])
        r=self.audit();self.assertEqual(r["summary"]["exact_duplicate_count"],0)
        self.assertEqual(r["summary"]["consumed_png_files_verified"],0)

    def test_no_same_key_is_scoped_not_global_claim(self):
        e=self.release["teacher_demonstrations"][0];e["key"]="different/0"
        self.change("metadata.json",lambda d:d.update(key="different/0"))
        r=self.audit()
        self.assertEqual(r["summary"]["counts"]["NO_SAME_KEY_OLD_EPISODE"],3)
        self.assertEqual(r["summary"]["deduplicated_new_count_within_same_key_scope"],3)
        self.assertIn("SAME task/key",r["scope"])

    def test_source_tamper_rejected(self):
        e=self.release["teacher_demonstrations"][0]
        (Path(e["branch"])/"actions.json").write_text("[]")
        with self.assertRaises(ValueError):self.audit()

    def test_old_pose_tamper_rejected(self):
        p=self.old/"train_pose.npy";a=np.load(p);a[0,0,0]+=.01;np.save(p,a)
        with self.assertRaises(ValueError):self.audit()

    def test_effective_index_tamper_rejected(self):
        self.release["teacher_demonstrations"][0]["valid_window_indices"]=[6,8,21];self.sync()
        with self.assertRaises(ValueError):self.audit()

    def test_new_repeat_rejected(self):
        self.change("metadata.json",lambda d:d.update(verification_only=True))
        with self.assertRaises(ValueError):self.audit()

    def test_image_symlink_rejected(self):
        e=self.release["teacher_demonstrations"][0];p=Path(e["branch"])/"rgb_0000.png"
        target=self.root/"external.png";target.write_bytes(p.read_bytes());p.unlink();p.symlink_to(target)
        with self.assertRaises(ValueError):self.audit()

    def test_old_samekey_allrows_compared_not_teacher_or_current(self):
        r=self.audit()
        self.assertEqual(r["episodes"][0]["old_same_key_valid_rows"],3)
        self.assertEqual(r["summary"]["counts"]["nonimage_candidate_pairs"],3)

    def test_old_episode_audit_requires_prior_trusted_sha(self):
        p=self.old/"index/train_episodes_audit.json"
        d=json.loads(p.read_text());d["stats"][0]["source_sha256"]["observations.json"]="0"*64
        save(p,d)
        with self.assertRaisesRegex(ValueError,"source SHA mismatch.*train_episodes_audit"):
            self.audit()

    def test_trust_anchor_tamper_rejected(self):
        save(self.anchor,dict(status="TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY",source_hashes={}))
        with self.assertRaisesRegex(ValueError,"source SHA mismatch"):self.audit()

    def test_absolute_time_and_index_not_fingerprint(self):
        root=Path(self.release["teacher_demonstrations"][0]["branch"])
        obs=json.loads((root/"observations.json").read_text())
        times=np.arange(len(obs),dtype=np.float64)*.0625
        cmds=np.tile(np.array([.1,0.,0.,.625],dtype=np.float32),(len(obs)-1,1))
        pose=np.zeros((7,4),dtype=np.float32);history=np.array([0,0,0,6])
        old=mod.scalar_parts(obs,times,cmds,history,pose)
        newobs=[copy.deepcopy(obs[0])]+copy.deepcopy(obs)
        newtimes=np.r_[times[0]-.0625,times]+32.
        newcmds=np.concatenate([cmds[:1],cmds])
        new=mod.scalar_parts(newobs,newtimes,newcmds,history+1,pose)
        self.assertEqual(old,new)

    def test_relative_time_is_part_of_consumption(self):
        root=Path(self.release["teacher_demonstrations"][0]["branch"])
        obs=json.loads((root/"observations.json").read_text());times=np.arange(len(obs),dtype=float)*.0625
        cmds=np.zeros((len(obs)-1,4),dtype=np.float32);pose=np.zeros((7,4),dtype=np.float32);h=np.array([0,0,0,6])
        a=mod.scalar_parts(obs,times,cmds,h,pose);times[0]-=.125
        b=mod.scalar_parts(obs,times,cmds,h,pose)
        self.assertNotEqual(a["times"],b["times"])

    def test_bbox_metadata_not_filter_actual_template_is(self):
        p=self.root/"uniform.png";Image.new("RGB",(8,8),(50,50,50)).save(p)
        pins=builder.Pins();reader=mod.ImageTensors(pins);inv={p.name:builder.file_sha(p)}
        self.assertEqual(reader.one(self.root,p.name,inv,[0,0,8,8]),
                         reader.one(self.root,p.name,inv,[1,1,7,7]))

    def test_exact_not_tolerance_equality(self):
        a=np.array([1.],dtype=np.float32);b=np.nextafter(a,np.float32(2))
        self.assertNotEqual(mod.tensor_digest(a),mod.tensor_digest(b))

    def test_nan_and_nonfloat32_rejected(self):
        with self.assertRaises(ValueError):mod.tensor_digest(np.array([np.nan],dtype=np.float32))
        with self.assertRaises(ValueError):mod.tensor_digest(np.array([1.],dtype=np.float64))


if __name__=="__main__":unittest.main()
