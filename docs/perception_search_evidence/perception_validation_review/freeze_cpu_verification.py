"""Run the local CPU suite and create a new verification record without overwrite."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from generate_val4_protocol import file_sha, generate, read, require


def main():
    root = Path(__file__).resolve().parent
    output, log = root/'VERIFICATION.json', root/'cpu_tests.log'
    require(not output.exists() and not log.exists(), 'refusing to overwrite CPU verification evidence')
    frozen_path = root/'permanent_val4_perception_protocol_v1.json'
    frozen = read(frozen_path)
    require(generate() == frozen, 'frozen protocol differs from current code or inputs')
    checked = []
    for path in sorted(root.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 9))
        checked.append(path.name)
    process = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root),
                              '-p', 'test_*.py', '-v'], cwd=str(root), text=True, capture_output=True)
    combined = process.stdout+process.stderr
    with log.open('x', encoding='utf-8') as stream: stream.write(combined)
    match = re.search(r'Ran (\d+) tests?', combined)
    require(process.returncode == 0 and match is not None, 'CPU verification failed; preserved test log')
    require(int(match.group(1)) == 21, 'unexpected CPU test denominator')
    artifacts = {path.name:{'sha256':file_sha(path),'bytes':path.stat().st_size}
                 for path in sorted(root.iterdir()) if path.is_file()}
    result = dict(stage='permanent_val4_perception_design_cpu_verification_v1',
        status='passed_cpu_design_tests_pending_remote_preflight_and_real_val_collector',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        test_count=int(match.group(1)), test_exit_code=process.returncode,
        python=sys.executable, python_version=sys.version,
        python39_ast_checked_files=checked,
        protocol_sha256=frozen['protocol_sha256'], protocol_file_sha256=file_sha(frozen_path),
        case_denominator=4, label_observation_denominator=229, prediction_denominator=225,
        gpu_collection_started=False, real_val_labels_generated=False,
        independent_raw_val_admission=False, optimizer_started=False,
        optimizer_input_allowed=False, formal_training_eligible=False,
        test_locked_read=False, frozen_train_bundle_modified=False,
        artifacts=artifacts)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','test_count','protocol_sha256',
        'protocol_file_sha256','gpu_collection_started')}, indent=2))


if __name__ == '__main__':
    main()
