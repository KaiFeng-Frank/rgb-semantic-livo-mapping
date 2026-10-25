"""Package compact reproducibility evidence; exclude raw datasets and predictions."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import digest, save

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
save(out/'execution.json', dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
     run_url=f'https://github.com/{os.environ.get("GITHUB_REPOSITORY")}/actions/runs/{os.environ.get("GITHUB_RUN_ID")}',
     packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()))
assets = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
          for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['manifest.json', 'SHA256SUMS', 'release-notes.md']]
save(out/'manifest.json', dict(assets=assets, raw_data_included=False))
(out/'SHA256SUMS').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in assets))
(out/'release-notes.md').write_text('''Offline seq07 loop-closure feasibility experiment.

Frozen FAST-LIVO2 trajectory and archived v0.6 B0 seed-1 pass-1 predictions.
Scan Context-style NumPy retrieval adapter, Open3D point-to-plane ICP, GTSAM Pose3 optimization, complete semantic-map replay.
Official KITTI odometry ground truth is read only by the separate evaluator.
See protocol.json, loops.json, baseline-parity.json and comparison.json for outcome and validation.
This archive may contain partial diagnostic outputs if the workflow failed; check execution.json's run URL.
Raw KITTI data and prediction caches are not republished here.
''')
