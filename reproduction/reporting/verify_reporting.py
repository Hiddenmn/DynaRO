"""Verify the reporting component without numerical libraries or model execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'reporting_manifest.json').read_text(encoding='utf-8'))
    if manifest['format'] != 'dynaro-saved-reporting-component-v1':
        raise ValueError('Unexpected reporting manifest')
    for name, item in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid reporting payload path')
        if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Changed reporting payload: {name}')
    return {'status': 'verified', 'payload_files': len(manifest['files']),
            'figures': 8, 'tables': 7, 'models_loaded': 0,
            'scope': 'Byte verification only; no array loading, training or evaluation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
