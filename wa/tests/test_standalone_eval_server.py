"""Bounded CPU tests: no model weights, CUDA operations, simulation or jobs."""
import ast
import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from wa.wm import eval_server as original
from wa.wm import standalone_eval_server as server


def fake_provenance(args):
    return dict(
        schema=server.direct.SCHEMA,
        status="DIRECT_CHECKPOINT_LOADED_NOT_OUTPUT_EQUIVALENCE",
        checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_step=args.checkpoint_step, kind="jepa", contract=server.direct.CONTRACT,
        encoder_sha256=server.HASHES["encoder"], initialization_weights_read=[],
        wla_initialization_loaded=False, jepa_initialization_loaded=False,
        optimizer_used=False, text_used=False, world_predictor_inference=False,
        output_equivalence_verified=False, closed_loop_verified=False,
        world_modules_retained=True, state_tensors_loaded=494, state_bytes_loaded=1452579160,
        source_files={"src/md_wla/fixture.py": "b" * 64},
    )


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ready = Path(self.tmp.name) / "ready.json"
        self.argv = ["--root", "/source/root", "--encoder-weight", "/weights/dino.pt",
                     "--wla-source", "/source/wla", "--checkpoint", "/weights/wa.pt",
                     "--checkpoint-sha256", "a" * 64, "--checkpoint-step", "59716",
                     "--mode", "mixed", "--noise-mode", "zero", "--ready", str(self.ready)]
        self.args = server.parse_args(self.argv)

    def metadata(self):
        model = types.SimpleNamespace(provenance=fake_provenance(self.args))
        value = server.metadata(model, self.args, "fixture GPU, not execution")
        value["url"] = "http://127.0.0.1:12345"
        return value

    def validate(self, value):
        server.validate_standalone_ready(value, checkpoint_sha="a" * 64,
                                         step=59716, mode="mixed")

    def test_original_session_is_reused_not_copied(self):
        self.assertIs(server.Session, original.Session)
        self.assertEqual(self.args.checkpoint_step, 59716)
        self.assertFalse(hasattr(self.args, "wla_checkpoint"))

    def test_explicit_identity_required_and_legacy_weight_argument_rejected(self):
        for flag in ("--checkpoint-sha256", "--checkpoint-step"):
            values = list(self.argv)
            pos = values.index(flag)
            del values[pos:pos + 2]
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    server.parse_args(values)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            server.parse_args(self.argv + ["--wla-checkpoint", "/old/unused.pt"])
        for flag, replacement in (("--checkpoint-sha256", "bad"),
                                  ("--checkpoint-step", "0")):
            values = list(self.argv)
            values[values.index(flag) + 1] = replacement
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                server.parse_args(values)

    def test_no_cluster_developer_check_and_no_existing_ready_overwrite(self):
        with patch.dict(server.os.environ, {"MD_AK_JOB_ID": "123"}):
            with self.assertRaisesRegex(ValueError, "no cluster smoke"):
                server.parse_args(self.argv + ["--developer-check"])
        self.ready.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            server.parse_args(self.argv)
        with self.assertRaises(FileExistsError):
            server.write_ready(self.ready, {"changed": True})
        self.assertEqual(self.ready.read_text(), "keep")
        self.assertEqual(list(self.ready.parent.iterdir()), [self.ready])

    def test_atomic_ready_is_closed_complete_json_at_publication(self):
        payload = {"sources": {str(i): "a" * 64 for i in range(288)}}
        real_dump, real_link = server.json.dump, server.os.link
        streams = []
        def dump(value, stream, **kwargs):
            self.assertFalse(self.ready.exists())
            streams.append(stream)
            return real_dump(value, stream, **kwargs)
        def link(source, destination):
            self.assertEqual(Path(source).parent, self.ready.parent)
            self.assertEqual(Path(destination), self.ready)
            self.assertTrue(streams[0].closed)
            self.assertFalse(self.ready.exists())
            self.assertEqual(json.loads(Path(source).read_text()), payload)
            real_link(source, destination)
            self.assertEqual(json.loads(self.ready.read_text()), payload)
        with patch.object(server.json, "dump", side_effect=dump), \
             patch.object(server.os, "link", side_effect=link) as publish:
            server.write_ready(self.ready, payload)
        publish.assert_called_once()
        self.assertEqual(list(self.ready.parent.iterdir()), [self.ready])

    def test_partial_write_failure_leaves_no_ready_or_temporary_file(self):
        def broken_dump(payload, stream, **kwargs):
            stream.write('{"partial":')
            stream.flush()
            raise OSError("simulated write failure")
        with patch.object(server.json, "dump", side_effect=broken_dump), \
             patch.object(server.os, "link") as publish:
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                server.write_ready(self.ready, {"wanted": True})
        publish.assert_not_called()
        self.assertEqual(list(self.ready.parent.iterdir()), [])

    def test_fsync_and_serialization_failures_cleanup_owned_temporary(self):
        with patch.object(server.os, "fsync", side_effect=OSError("fsync failed")), \
             patch.object(server.os, "link") as publish:
            with self.assertRaisesRegex(OSError, "fsync failed"):
                server.write_ready(self.ready, {"wanted": True})
        publish.assert_not_called()
        self.assertEqual(list(self.ready.parent.iterdir()), [])
        with self.assertRaises(ValueError):
            server.write_ready(self.ready, {"invalid": float("nan")})
        self.assertEqual(list(self.ready.parent.iterdir()), [])

    def test_competing_ready_winner_is_not_overwritten_and_temp_is_removed(self):
        real_link = server.os.link
        def race(source, destination):
            self.ready.write_text("other attempt")
            return real_link(source, destination)
        with patch.object(server.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                server.write_ready(self.ready, {"wanted": True})
        self.assertEqual(self.ready.read_text(), "other attempt")
        self.assertEqual(list(self.ready.parent.iterdir()), [self.ready])

    def test_broken_symlink_ready_rejected(self):
        self.ready.symlink_to(self.ready.parent / "missing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            server.parse_args(self.argv)

    def test_ready_keeps_original_top_level_contract_and_validates_provenance(self):
        ready = self.metadata()
        self.validate(ready)
        self.assertEqual(ready["sampling_steps"], 4)
        self.assertEqual(ready["seed"], "7+step")
        self.assertEqual(set(ready["standalone"]["source_sha256"]), {"session", "loader", "rpc"})
        self.assertEqual(ready["standalone"]["upstream_source_commits"], server.PINS)
        self.assertNotIn("metadata", ready)
        self.assertFalse(ready["world_predictor_inference"])

    def test_wrong_identity_seed_mode_source_or_upstream_rejected(self):
        mutations = [
            lambda r: r.update(checkpoint_sha256="c" * 64),
            lambda r: r.update(step=59715),
            lambda r: r.update(mode="image"),
            lambda r: r.update(seed="8+step"),
            lambda r: r.pop("standalone"),
            lambda r: r["standalone"]["source_sha256"].update(loader="c" * 64),
            lambda r: r["standalone"]["upstream_source_commits"].update(dinov2="foreign"),
            lambda r: r["standalone"].update(extra="not permitted"),
        ]
        for i, mutation in enumerate(mutations):
            ready = self.metadata()
            mutation(ready)
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.validate(ready)

    def test_foreign_or_false_direct_provenance_rejected(self):
        values = {"checkpoint": "/other.pt", "checkpoint_sha256": "0" * 64,
                  "checkpoint_step": 59715, "contract": "wrong",
                  "initialization_weights_read": ["/old.pt"],
                  "wla_initialization_loaded": True, "jepa_initialization_loaded": True,
                  "world_modules_retained": False, "text_used": 0,
                  "output_equivalence_verified": True, "closed_loop_verified": True,
                  "source_files": {}, "state_tensors_loaded": True, "state_bytes_loaded": 0}
        for key, value in values.items():
            ready = self.metadata()
            ready["standalone"]["provenance"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(ready)

    def test_nonlocal_url_or_invalid_port_rejected(self):
        for url in ("http://0.0.0.0:123", "http://example:123", "http://127.0.0.1:0",
                    "http://127.0.0.1:65536", "http://127.0.0.1:123/predict", None):
            ready = self.metadata()
            ready["url"] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.validate(ready)

    def test_developer_requests_match_original_template_history_and_uwb(self):
        for mode in ("image", "mixed"):
            requests = []
            session = types.SimpleNamespace(predict=lambda r: requests.append(copy.deepcopy(r)) or {"finite": 1})
            first, second = server.developer_check(session, mode)
            self.assertEqual(first, {"finite": 1})
            self.assertEqual(second, {"finite": 1})
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["initial_bbox"], [20, 20, 120, 200])
            self.assertIsNone(requests[1]["initial_bbox"])
            self.assertEqual([r["timestamp_s"] for r in requests], [0., .05])
            self.assertEqual(requests[0]["rgb_png"], requests[1]["rgb_png"])
            expected = {"rgb_png", "timestamp_s", "initial_bbox"} | ({"uwb"} if mode == "mixed" else set())
            self.assertEqual(set(requests[0]), expected)
            if mode == "mixed":
                self.assertEqual(requests[0]["uwb"], dict(
                    polar=[2., .1], timestamp_s=0., valid=True, source="ideal_simulated_uwb",
                    range_noise_m=0., bearing_noise_rad=0., delay_s=0.))

    def test_main_calls_only_direct_loader_and_cpu_mocked_developer_branch(self):
        model = MagicMock()
        model.provenance = fake_provenance(self.args)
        model.cuda.return_value = model
        model.eval.return_value = model
        model.requires_grad_.return_value = model
        session = MagicMock()
        session.predict.return_value = {"finite": 1}
        with patch.dict(server.os.environ, {}, clear=True), \
             patch.object(server.direct, "load_standalone", return_value=model) as load, \
             patch.object(server.torch, "set_num_threads") as threads, \
             patch.object(server.torch, "manual_seed") as seed, \
             patch.object(server.torch.cuda, "set_device") as device, \
             patch.object(server.torch.cuda, "get_device_name", return_value="CPU mock"), \
             patch.object(server, "Session", return_value=session), \
             patch.object(server, "HTTPServer") as http:
            server.main(self.argv + ["--developer-check"])
        load.assert_called_once_with("/source/root", "/weights/dino.pt", "/source/wla",
                                     "/weights/wa.pt", "a" * 64, 59716)
        threads.assert_called_once_with(2)
        seed.assert_called_once_with(7)
        device.assert_called_once_with(0)
        http.assert_not_called()
        model.requires_grad_.assert_called_once_with(False)
        result = json.loads(self.ready.read_text())
        self.assertEqual(result["status"], "DEVELOPER_INTERFACE_PASS_NOT_BENCHMARK")
        self.assertIn("standalone", result["metadata"])
        self.assertNotIn("SR", result)

    def test_handler_preserves_reset_predict_and_error_protocol(self):
        session = MagicMock()
        session.predict.side_effect = lambda data: {"received": data}
        handler = server.handler_class(session)
        def request(path, data):
            value = object.__new__(handler)
            encoded = json.dumps(data).encode()
            value.path = path
            value.headers = {"Content-Length": str(len(encoded))}
            value.rfile = io.BytesIO(encoded)
            value.wfile = io.BytesIO()
            value.send_response = MagicMock()
            value.send_header = MagicMock()
            value.end_headers = MagicMock()
            with patch.object(server.traceback, "print_exc"):
                value.do_POST()
            return value.send_response.call_args.args[0], json.loads(value.wfile.getvalue())
        self.assertEqual(request("/reset", {}), (200, {"reset": True}))
        session.reset.assert_called_once_with()
        self.assertEqual(request("/predict", {"rgb_png": "fixture"}), (200, {"received": {"rgb_png": "fixture"}}))
        self.assertEqual(request("/reset", {"invalid": True})[0], 500)
        self.assertEqual(request("/other", {})[0], 500)
        session.predict.side_effect = ValueError("invalid input")
        self.assertEqual(request("/predict", {}), (500, {"error": "invalid input"}))


class LauncherTests(unittest.TestCase):
    def test_only_explicit_server_and_admission_deltas_to_old_launcher(self):
        root = Path(server.__file__).parent
        original_text = (root / "eval_full_mixed_launch.py").read_text()
        candidate = (root / "eval_full_standalone_launch.py").read_text()
        changes = [
            ("from wa.wm.student_eval_partition import task_scope,write_partition",
             "from wa.wm.student_eval_partition import task_scope,write_partition\nfrom wa.wm.standalone_eval_server import validate_standalone_ready"),
            ("contract=model_contract(os.environ)",
             "contract=model_contract(os.environ)\nif not contract:raise ValueError('standalone launcher requires explicit student identity')"),
            ("'-m','wa.wm.eval_server'", "'-m','wa.wm.standalone_eval_server'"),
            ("'--wla-source',str(ROOT/'dependencies/wla_v1'),'--wla-checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt',",
             "'--wla-source',str(ROOT/'dependencies/wla_v1'),\n        '--checkpoint-sha256',contract['checkpoint_sha'],'--checkpoint-step',str(contract['step']),"),
            ("        validate_ready(json.loads(ready.read_text()),mode=mode,**contract)",
             "        validate_ready(json.loads(ready.read_text()),mode=mode,**contract)\n        validate_standalone_ready(json.loads(ready.read_text()),mode=mode,**contract)"),
        ]
        expected = original_text
        for old, new in changes:
            self.assertEqual(expected.count(old), 1)
            expected = expected.replace(old, new)
        self.assertEqual(candidate, expected)
        # AST parse only: importing either launcher would start real workers.
        ast.parse(candidate)
        self.assertNotIn("--wla-checkpoint", candidate)
        self.assertNotIn("mz_jepa", candidate)

    def test_server_argv_has_explicit_sha_step_and_original_lane_options(self):
        text = (Path(server.__file__).parent / "eval_full_standalone_launch.py").read_text()
        tree = ast.parse(text)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "server" for t in n.targets)]
        self.assertEqual(len(calls), 1)
        expression = ast.Expression(calls[0].value.args[0])
        arguments = eval(compile(expression, "<argv-only>", "eval"), {"str": str}, dict(
            ROOT=Path("/ROOT"), os=types.SimpleNamespace(environ={"WA_DIAG_CHECKPOINT": "/wa.pt"}),
            ready=Path("/OUT/shard_00/server_ready.json"), mode="mixed",
            contract={"checkpoint_sha": "a" * 64, "step": 59716}))
        self.assertEqual(arguments[:4], ["/ROOT/probe_env/bin/python", "-u", "-m",
                                         "wa.wm.standalone_eval_server"])
        options = dict(zip(arguments[4::2], arguments[5::2]))
        self.assertEqual(options["--checkpoint"], "/wa.pt")
        self.assertEqual(options["--checkpoint-sha256"], "a" * 64)
        self.assertEqual(options["--checkpoint-step"], "59716")
        self.assertEqual(options["--noise-mode"], "zero")
        self.assertEqual(options["--mode"], "mixed")
        self.assertEqual(options["--ready"], "/OUT/shard_00/server_ready.json")
        self.assertNotIn("--wla-checkpoint", options)


if __name__ == "__main__":
    unittest.main()
