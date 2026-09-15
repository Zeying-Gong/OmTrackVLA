# Clean48 runtime navigation cache inventory v1

All 48 actual runtime navigation caches exist and load successfully with CPU-only Habitat `PathFinder`. They total **1,827,996 bytes**. Every actual cache differs in path and content hash from the `.basis.navmesh` file beside its scene geometry. In **41 of 48 scenes**, the two files also report different numbers of connected navigation regions (islands).

This closes a dependency gap in `clean48_asset_inventory_v1.json`: that earlier inventory checked the adjacent mesh as a derived candidate, while the simulator explicitly loads a separate cache. It does not retroactively add cache pins to the earlier V1/V2 collection plans.

## Actual path selection

The frozen V2 configuration uses dataset type `PointNav-v1` and `scenes_dir=data/scene_datasets`. `PointNavDatasetV1.from_json` strips an existing `data/scene_datasets/` prefix, then joins the scene ID to `scenes_dir`. `RearrangeSim-v2._load_navmesh` takes the first two components of this resolved runtime scene ID, adds `navmeshes`, and uses the basename before its first period.

For the first sample:

| Item | Path or value |
| --- | --- |
| Original dataset scene ID | `hm3d/train/00402-zR6kPe1PsyS/zR6kPe1PsyS.basis.glb` |
| Runtime scene ID | `data/scene_datasets/hm3d/train/00402-zR6kPe1PsyS/zR6kPe1PsyS.basis.glb` |
| Actual runtime cache | `data/scene_datasets/navmeshes/zR6kPe1PsyS.navmesh` |
| Actual cache resolved path | `/data/nfs/share/evt_bench/navmeshes/zR6kPe1PsyS.navmesh` |
| Actual cache SHA256 | `32c18267379578ee878c82477d2eac80ad987364dd78ebdf72f96bd9a2bf7708` |
| Actual cache bytes / islands / area | 21,356 / 5 / 33.73324 m² |
| Adjacent candidate | `data/scene_datasets/hm3d/train/00402-zR6kPe1PsyS/zR6kPe1PsyS.basis.navmesh` |
| Adjacent candidate SHA256 | `9b4a1d7853ca57957372cc67bcc26cfee68bc86800dd0a3186550533d368cefe` |
| Adjacent bytes / islands / area | 35,976 / 4 / 75.46180 m² |

All repository-relative paths are under `/data/nfs/share/wam_tracking/OmTrackVLA`. Scene-name case is preserved. Applying the cache formula directly to the original `hm3d/...` scene ID would produce the wrong cache directory.

## Settings and runtime behavior

All 48 runtime caches report the same 17 stored settings as the frozen V2 recomputation reference. Their agent radius is 0.3 m, agent height 1.5 m, maximum climb 0.2 m, maximum slope 45°, and `include_static_objects=True`. Floating-point storage explains values such as `0.30000001192092896` in the JSON.

All 48 adjacent meshes use radius 0.1 m. In 46 adjacent meshes, `include_static_objects=False`; the other two store `True`. The runtime cache has between 1 and 17 islands across these scenes. `inventory.json` records every island's area, the largest-island index, bounds, settings, file hash, resolved path, size, modification time, and inode for both sets of 48 meshes.

When the runtime cache exists, `_load_navmesh` calls `load_nav_mesh` on it. It does not compare stored settings with the current agent configuration or check the load method's boolean return. When the cache is absent, it creates default `NavMeshSettings`, overrides radius, height, climb and slope from `agent_0` (falling back to `main_agent`), sets `include_static_objects=True`, recomputes the mesh, creates the cache directory, and saves the result. That missing-cache branch would write a new runtime dependency. It was **not executed** by this inventory.

Although the surrounding code comments refer to indoor islands, this runtime calls `get_largest_island_index(..., allow_outdoor=True)`. The pinned helper sorts islands by area and selects the largest; it does not run the outdoor raycast classifier in that branch. This inventory computes the same largest-area choice using CPU PathFinder results.

## Verification and source pins

Remote execution used Python 3.9.23 and Habitat-Sim 0.3.1 with empty `CUDA_VISIBLE_DEVICES`, empty `PYTHONPATH`, disabled bytecode writes, and one thread for OMP/MKL/OpenBLAS. The probe created `PathFinder` objects only. It created no Simulator, renderer, GPU model, collection episode, or optimizer. It did not save or recompute any navigation mesh. All 96 mesh hashes and stat records, the runtime source files, and the installed PathFinder implementation references were unchanged after the probe.

The offline verifier passed **1,140 checks**. It independently derived all 48 runtime/adjacent paths, checked exact sample coverage and source pins, verified the parsed report against the raw remote log, checked all island counts and area summaries, reproduced settings differences, and matched all adjacent file references against the preceding asset inventory. The authoritative remote exit code was 0.

| Source or evidence | SHA256 |
| --- | --- |
| `evt_bench/rearrange_sim_v2.py` | `e9827769fcb78a3344d45eb14c994b04e8fbf75a418474709d1d20a50aee84b6` |
| `habitat-lab/habitat/datasets/pointnav/pointnav_dataset.py` | `23c7c101c92107800d7b098a524d43dc3bf0cea87bbb5199c434d75c381e0a9e` |
| `habitat-lab/habitat/datasets/rearrange/navmesh_utils.py` | `61f7fabc08e12af9e3bbd2f273e1767c435be6a05f4056d69508d647c0e0f439` |
| Clean training candidate pool | `4bce60c6cc74351048101019d48a977d30f67669d7ab7b7d199ffd08582395af` |
| Preceding clean48 asset inventory | `25bbb97ba387872c08b01dbcf3c07d43e6388e1abe3d6260f1aabc2e2d4e6a10` |
| V2 frozen plan supplying reference configuration | `64b072f660fdc304e1119ffb4bf5eb1e495b88e66ed015403342c6f128ecebc9` |
| `inventory.json` | `69792a810cb37596b24baafae96b7895fd19ba71410edf22eefc7aaa1abebaeb` |
| `verification.json` | `9c4c7d44e3aa9c9817497cafa4be5bec7f81d371adbc08fd66bccb5648c1120c` |

`runtime_cache_table.csv` provides a compact scene-by-scene comparison. `request.json`, `inventory_execution.json`, the raw logs, and downloaded source copies preserve the derivation and execution evidence. Runtime Python/native PathFinder implementation hashes are in `inventory.json`.

## Consequences for subsequent runs

Future collection and evaluation plans should pin these actual runtime cache paths, resolved paths and byte hashes as required dependencies, and check them before environment creation. Adjacent meshes should retain their own identity. A missing or changed runtime cache should be detected before entering a branch that silently recomputes it. A deliberate recomputation would require new evidence and a new plan.

Stored settings matching V2 does not establish how or from which geometry a cache was originally generated. This is a snapshot of current cache bytes; earlier unpinned runs cannot be assigned those bytes retrospectively. The probe did not render the scenes, construct a new simulator configuration for every candidate, or validate that each target trajectory is reachable on the selected island. Teacher quality, navigation success, training eligibility, and benchmark readiness remain separate checks.
