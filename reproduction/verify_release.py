"""Check the assembled local package; optional plans never execute experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    if manifest['format'] != 'dynaro-integrated-local-candidate-v1' or manifest['published']:
        raise ValueError('Unexpected local-candidate identity')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(manifest['files']) | {'release_manifest.json'}:
        raise ValueError('Unexpected or missing release file')
    for name, entry in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Changed release file: {name}')
    return {'status': 'local_candidate_verified', 'files': len(manifest['files']),
            'components': len(manifest['components']), 'published': False,
            'research_model_execution': False}


def check_plans(root=ROOT):
    root = Path(root).resolve()
    core = str(root / 'code/model_core')
    inputs = str(root / 'inputs')
    weights = str(root / 'weights')
    evaluation = str(root / 'evaluation')
    training = str(root / 'training')
    commands = [('model_core_and_source_data', ['code/verify_candidate.py', '--root', str(root / 'code')])]
    for dataset in ('train25', 'dev25', 'test25', 'train50', 'test100'):
        commands.append(('input_' + dataset, ['inputs/reproduce_inputs.py', '--set', dataset]))
    for study in ('mixed_horizon/dynaro/20260811', 'learning_rule_25/state_prioritized/20260811'):
        shared = ['--core', core, '--inputs', inputs, '--model-key', study]
        commands.append(('frozen_' + study, ['evaluation/evaluate.py', *shared, '--weights', weights]))
        commands.append(('train_' + study, ['training/train.py', *shared, '--evaluation', evaluation]))
        commands.append(('retrained_' + study, ['retrained_evaluation/evaluate_retrained.py', *shared,
            '--weights', weights, '--evaluation', evaluation, '--training', training]))
    reports = []
    for name, command in commands:
        # -S removes third-party libraries; these plans must work without any
        # tensor/array/GPU package, and never receive an --execute argument.
        proc = subprocess.run([sys.executable, '-B', '-I', '-S', str(root / command[0]), *command[1:]],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace')
        if proc.returncode:
            raise RuntimeError(f'Metadata plan failed: {name}\n{proc.stdout}')
        json.loads(proc.stdout)
        reports.append({'plan': name, 'exit_code': proc.returncode})
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', action='store_true', help='Also run standard-library-only metadata plans')
    args = parser.parse_args()
    result = verify()
    if args.plans:
        result['metadata_plans'] = check_plans()
    print(json.dumps(result, indent=2))
