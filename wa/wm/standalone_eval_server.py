"""Standalone WA RPC with the original Session and no initialization-weight inputs.

This additive entry point does not replace the frozen eval_server. Its ready
record retains the original protocol and adds direct-loader/source provenance.
"""
import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import re
import tempfile
import traceback

from PIL import Image
import torch

from wa.wm import eval_server as original
from wa.wm import standalone_inference as direct
from wa.wm.full_mixed_contract import validate_checkpoint_identity, validate_ready
from wa.wm.loaders import HASHES, PINS, sha

Session = original.Session
SCHEMA = "wa.standalone_eval_server.v1"


def source_pins():
    return {name: sha(path) for name, path in (
        ("session", original.__file__),
        ("loader", direct.__file__),
        ("rpc", __file__),
    )}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "encoder-weight", "wla-source", "checkpoint",
                 "checkpoint-sha256", "ready"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--mode", choices=("image", "mixed"), default="image")
    parser.add_argument("--noise-mode", choices=("random", "zero"), default="random")
    parser.add_argument("--developer-check", action="store_true")
    args = parser.parse_args(argv)
    validate_checkpoint_identity(args.checkpoint_sha256, args.checkpoint_step)
    if args.developer_check and os.environ.get("MD_AK_JOB_ID"):
        raise ValueError("no cluster smoke")
    ready = Path(args.ready)
    if ready.exists() or ready.is_symlink():
        raise ValueError("ready destination already exists")
    return args


def metadata(model, args, gpu):
    result = dict(
        checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
        step=args.checkpoint_step, gpu=gpu, mode=args.mode,
        noise_mode=args.noise_mode, sampling_steps=4, seed="7+step",
        text_used=False, world_predictor_inference=False,
        standalone=dict(schema=SCHEMA, source_sha256=source_pins(),
                        upstream_source_commits=dict(PINS),
                        provenance=model.provenance),
    )
    return result


def validate_standalone_ready(ready, *, checkpoint_sha, step, mode="mixed"):
    """Extra admission in the additive launcher; original ready checks stay intact."""
    validate_ready(ready, checkpoint_sha=checkpoint_sha, step=step, mode=mode)
    if ready.get("seed") != "7+step":
        raise ValueError("wrong standalone seed")
    if not isinstance(ready.get("url"), str) or re.fullmatch(
            r"http://127\.0\.0\.1:[1-9][0-9]{0,4}", ready["url"]) is None:
        raise ValueError("standalone RPC must use the local loopback endpoint")
    port = int(ready["url"].rsplit(":", 1)[1])
    if port > 65535:
        raise ValueError("invalid standalone RPC port")
    declaration = ready.get("standalone")
    if not isinstance(declaration, dict) or set(declaration) != {
            "schema", "source_sha256", "upstream_source_commits", "provenance"}:
        raise ValueError("missing standalone provenance")
    if declaration["schema"] != SCHEMA or declaration["source_sha256"] != source_pins():
        raise ValueError("standalone RPC source mismatch")
    if declaration["upstream_source_commits"] != PINS:
        raise ValueError("standalone upstream source mismatch")
    p = declaration["provenance"]
    if not isinstance(p, dict):
        raise ValueError("missing direct-load provenance")
    expected = dict(
        schema=direct.SCHEMA, status="DIRECT_CHECKPOINT_LOADED_NOT_OUTPUT_EQUIVALENCE",
        checkpoint=ready["checkpoint"], checkpoint_sha256=checkpoint_sha,
        checkpoint_step=step, kind="jepa", contract=direct.CONTRACT,
        encoder_sha256=HASHES["encoder"], initialization_weights_read=[],
    )
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError("wrong standalone provenance " + key)
    for key in ("wla_initialization_loaded", "jepa_initialization_loaded",
                "optimizer_used", "text_used", "world_predictor_inference",
                "output_equivalence_verified", "closed_loop_verified"):
        if p.get(key) is not False:
            raise ValueError("wrong standalone provenance " + key)
    if p.get("world_modules_retained") is not True:
        raise ValueError("standalone world modules not retained")
    for key in ("state_tensors_loaded", "state_bytes_loaded"):
        if type(p.get(key)) is not int or p[key] <= 0:
            raise ValueError("invalid standalone state count")
    if not isinstance(p.get("source_files"), dict) or not p["source_files"]:
        raise ValueError("missing WLA source inventory")


def write_ready(path, payload):
    """Publish complete JSON atomically, without replacing another attempt."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=destination.parent,
                prefix="." + destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-parent hard link is atomic and fails if destination exists.
        # The closed, complete inode becomes visible in one operation.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def handler_class(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/reset":
                    if request:
                        raise ValueError("reset payload must be empty")
                    session.reset()
                    value = {"reset": True}
                elif self.path == "/predict":
                    value = session.predict(request)
                else:
                    raise ValueError("unknown endpoint")
                data = json.dumps(value, allow_nan=False).encode()
                self.send_response(200)
            except Exception as error:
                traceback.print_exc()
                data = json.dumps({"error": str(error)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def developer_check(session, mode):
    stream = io.BytesIO()
    Image.new("RGB", (224, 224)).save(stream, format="PNG")
    request = dict(rgb_png=base64.b64encode(stream.getvalue()).decode(),
                   timestamp_s=0., initial_bbox=[20, 20, 120, 200])
    if mode == "mixed":
        request["uwb"] = dict(polar=[2., .1], timestamp_s=0., valid=True,
                              source="ideal_simulated_uwb", range_noise_m=0.,
                              bearing_noise_rad=0., delay_s=0.)
    first = session.predict(request)
    request.update(timestamp_s=.05, initial_bbox=None)
    second = session.predict(request)
    return first, second


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.cuda.set_device(0)
    model = direct.load_standalone(
        args.root, args.encoder_weight, args.wla_source, args.checkpoint,
        args.checkpoint_sha256, args.checkpoint_step,
    )
    model.cuda().eval().requires_grad_(False)
    session = Session(model, args.mode, args.noise_mode)
    ready = metadata(model, args, torch.cuda.get_device_name(0))
    if args.developer_check:
        first, second = developer_check(session, args.mode)
        write_ready(args.ready, dict(
            status="DEVELOPER_INTERFACE_PASS_NOT_BENCHMARK",
            metadata=ready, first=first, second=second))
        return
    server = HTTPServer(("127.0.0.1", 0), handler_class(session))
    try:
        ready["url"] = "http://127.0.0.1:" + str(server.server_port)
        write_ready(args.ready, ready)
        print("WA_SERVER_READY", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
