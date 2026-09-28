# WA-Mobile portable experiment workspace

This directory is independent of the inherited OmTrackVLA training scripts.
Status: infrastructure only; there is no WA training entry point yet.
Python >=3.10; result tooling uses only the standard library.
PyTorch/CUDA are optional for CPU tests, required for GPU readiness inspection.

## Both execution lanes
Run from the repository root after checking out `wa` at a recorded commit.
Use a NEW persistent output directory outside this Git checkout.

```bash
python3 -m unittest discover -s wa/tests -v
python3 wa/tools/exchange.py preflight --lane managed --output /persistent/runs/wa-001
# External machine: select this lane to require exactly 8 visible H100 GPUs.
python3 wa/tools/exchange.py preflight --lane external-h100 --output /persistent/runs/wa-ext-001
```

Preflight records Git revision/dirty file names, Python, platform, PyTorch,
CUDA and visible GPU names. It does not certify scheduler, data or model readiness.
No environment variables, credentials or Git remote URLs are collected.
Managed lane records actual GPU models: a cluster label alone is insufficient.
A800/4090 use the platform K8s scheduler; Aliyun uses Ray. Scheduler manifests
will be prepared for the concrete approved experiment, not guessed here.

## Return results through GitHub
A future runner writes `metrics.json` and `config.json` beside `environment.json`.
Use JSON null for unavailable metrics, never fabricate zero-valued results.
A metrics report must include status, protocol_id, dataset_sha256, seed,
and per-mode/per-scenario metrics with episode counts and units.

```bash
python3 wa/tools/exchange.py pack --run /persistent/runs/wa-ext-001 --output /persistent/wa-ext-001.zip
python3 wa/tools/exchange.py verify --bundle /persistent/wa-ext-001.zip
```

Upload the ZIP as a GitHub Release asset in this repository and share its URL.
The assistant can retrieve it and verify hashes before analysis. Preflight-only
bundles are explicitly not experimental results. Only three allowlisted JSON files
are packed, at most 2 MiB each; raw data and checkpoints are never included.
Review their contents before uploading. Do not store credentials in config/metrics.
Compare matching protocol, dataset, seed, effective batch and precision; hardware
and numerical differences must be reported separately.
