"""Versioned MP3D PLY orientation repair; no shared assets/runtime mutation.

Habitat-Sim's PLY importer already converts -Z gravity to -Y. The stage
must not apply the GLB's Z-up frame again to that semantic mesh.
TrackEnv reads episode.scene_dataset_config, overriding the outer YAML.
"""
import copy
import hashlib
from pathlib import Path

VERSION = "mp3d_semantic_ply_v1"
CONFIG = Path(__file__).resolve().parents[1] / "config" / (VERSION + ".scene_dataset_config.json")

def is_mp3d_ply_scene(scene_id):
    path = Path(scene_id)
    return ("mp3d" in path.parts and path.suffix == ".glb"
            and path.parent.name == path.stem)

def prepare_episode(episode, asset_root):
    """Copy, never mutate manifest episode; reject unreviewed custom configs."""
    result = copy.deepcopy(episode)
    if not is_mp3d_ply_scene(episode.scene_id):
        return result
    if episode.scene_dataset_config != "default":
        raise ValueError("MP3D repair requires audited default scene dataset")
    scene = Path(episode.scene_id)
    if not scene.is_absolute():
        scene = Path(asset_root) / scene
    for path in (scene, scene.with_name(scene.stem + "_semantic.ply")):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    result.scene_dataset_config = str(CONFIG)
    return result

def validate_runtime(sim, episode):
    """Fail before inference if the episode-level repair did not take effect."""
    if not is_mp3d_ply_scene(episode.scene_id):
        return
    if episode.scene_dataset_config != str(CONFIG):
        raise ValueError("MP3D episode did not select repaired config")
    attrs = sim.get_stage_initialization_template()
    expected = dict(orient_up=(0,0,1), orient_front=(0,1,0),
                    semantic_orient_up=(0,1,0), semantic_orient_front=(0,0,-1))
    for key, value in expected.items():
        if tuple(getattr(attrs, key)) != value:
            raise ValueError("unexpected stage orientation: " + key)
    if not attrs.semantic_asset_handle.endswith("_semantic.ply"):
        raise ValueError("repair expects MP3D semantic PLY")
    if attrs.render_asset_handle != attrs.collision_asset_handle:
        raise ValueError("unreviewed MP3D render/collision asset combination")

def provenance():
    return {"version": VERSION, "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "scope": "MP3D GLB + semantic PLY with default episode scene config",
            "changes": "semantic stage orientation only; RGB/collision Z-up retained",
            "requires_fresh_metrics": True}
