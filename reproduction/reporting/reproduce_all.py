"""Reproduce all eight manuscript figures and seven tables from saved outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from verify_reporting import verify
COMMANDS = (
    ('quantitative', 'reproduce.py'), ('sampling', 'reproduce_sampling.py'),
    ('architecture', 'reproduce_architecture.py'), ('central64', 'reproduce_3d.py'),
    ('parent128', 'reproduce_parent3d.py'), ('train_case', 'reproduce_train3d.py'),
)


def run(output):
    output = Path(output).resolve()
    verify(ROOT)
    if sys.flags.optimize:
        raise ValueError('Do not use -O')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if output == ROOT or ROOT.is_relative_to(output) or output.is_relative_to(ROOT):
        raise ValueError('Place outputs outside the immutable reporting component')
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    for name, script in COMMANDS:
        result = subprocess.run([sys.executable, '-B', '-I', str(ROOT / script),
            '--output-dir', str(output / name)], env=env, cwd=ROOT, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        (output / f'{name}.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {name}.log in the chosen output directory')
        print(f'Completed {name}', flush=True)
    report = {'status': 'complete_saved_output_reproduction', 'figures': 8, 'tables': 7,
              'new_training': False, 'new_inference': False, 'new_simulation': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
