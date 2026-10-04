import copy, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from wa.wm.semantic_scene import CONFIG, prepare_episode, is_mp3d_ply_scene, provenance, validate_runtime

class SemanticSceneTests(unittest.TestCase):
    def test_config_preserves_render_frame(self):
        d = json.loads(CONFIG.read_text())["stages"]["default_attributes"]
        self.assertEqual(d, dict(up=[0,0,1], front=[0,1,0],
                                semantic_up=[0,1,0], semantic_front=[0,0,-1]))
    def test_mp3d_only(self):
        self.assertTrue(is_mp3d_ply_scene("data/mp3d/abc/abc.glb"))
        for p in ["data/hm3d/abc/abc.basis.glb", "data/mp3d/abc/other.glb", "data/abc/abc.glb"]:
            self.assertFalse(is_mp3d_ply_scene(p))
    def test_copy_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            scene=Path(td)/"mp3d/abc/abc.glb";scene.parent.mkdir(parents=True)
            scene.touch();scene.with_name("abc_semantic.ply").touch()
            ep=SimpleNamespace(scene_id=str(scene),scene_dataset_config="default",
                start_position=[1,2,3],info={"main_human_semantic_id":1098})
            old=copy.deepcopy(vars(ep));fixed=prepare_episode(ep,td)
            self.assertEqual(vars(ep),old)
            self.assertEqual(fixed.scene_dataset_config,str(CONFIG))
            self.assertEqual({k:v for k,v in vars(fixed).items() if k!="scene_dataset_config"},
                             {k:v for k,v in old.items() if k!="scene_dataset_config"})
    def test_non_mp3d_unchanged(self):
        ep=SimpleNamespace(scene_id="hm3d/abc/abc.basis.glb",scene_dataset_config="custom")
        self.assertEqual(vars(prepare_episode(ep,".")),vars(ep))
    def test_custom_rejected(self):
        ep=SimpleNamespace(scene_id="mp3d/abc/abc.glb",scene_dataset_config="custom")
        with self.assertRaises(ValueError):prepare_episode(ep,".")
    def test_missing_asset_rejected(self):
        ep=SimpleNamespace(scene_id="mp3d/abc/abc.glb",scene_dataset_config="default")
        with self.assertRaises(FileNotFoundError):prepare_episode(ep,"/nonexistent")
    def test_provenance(self):
        self.assertTrue(provenance()["requires_fresh_metrics"])
        self.assertEqual(len(provenance()["config_sha256"]),64)

    def test_runtime_frame_guard(self):
        ep=SimpleNamespace(scene_id="mp3d/abc/abc.glb",scene_dataset_config=str(CONFIG))
        attrs=SimpleNamespace(orient_up=(0,0,1),orient_front=(0,1,0),
            semantic_orient_up=(0,1,0),semantic_orient_front=(0,0,-1),
            semantic_asset_handle="abc_semantic.ply",render_asset_handle="abc.glb",
            collision_asset_handle="abc.glb")
        sim=SimpleNamespace(get_stage_initialization_template=lambda:attrs)
        validate_runtime(sim,ep)
        attrs.semantic_orient_up=(0,0,1)
        with self.assertRaises(ValueError):validate_runtime(sim,ep)
        attrs.semantic_orient_up=(0,1,0)
        ep.scene_dataset_config="default"
        with self.assertRaises(ValueError):validate_runtime(sim,ep)

if __name__=="__main__":unittest.main()
