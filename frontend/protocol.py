"""Validate bounded replay plans and the frozen estimator, before data or GT access."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVES = {'07': '2011_09_30_drive_0027', '09': '2011_09_30_drive_0033'}

def load(path=None):
    cfg = json.loads(Path(path or REPO/'frontend/protocol.json').read_text())
    part = cfg['short_sequence']
    seq, first, end = part['sequence'], part['first_frame'], part['end_exclusive']
    if seq not in DRIVES or first != 0 or not 0 < end <= (120 if seq == '07' else 240):
        raise ValueError('outside the admitted bounded replay windows')
    for relative, expected in cfg.get('frozen_sources', {}).items():
        if hashlib.sha256((REPO/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen estimator changed: '+relative)
    return cfg

def drive(cfg):
    return DRIVES[cfg['short_sequence']['sequence']]

def raw_dir(root, cfg):
    return root/'raw/2011_09_30'/f'{drive(cfg)}_sync'
