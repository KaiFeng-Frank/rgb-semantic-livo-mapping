import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from frontend.protocol import load
from frontend.assess_validation import assess

ap = argparse.ArgumentParser()
ap.add_argument('download', type=Path)
args = ap.parse_args()
archives = list(args.download.rglob('frontend-evidence.tar.gz'))
assert len(archives) == 1, archives
archive = archives[0]
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive.with_name('frontend-evidence.sha256').read_text().split()[0] == digest
out = args.download / 'extracted'
out.mkdir(exist_ok=True)
with tarfile.open(archive) as tf:
    for m in tf.getmembers():
        assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        assert m.isfile() or m.isdir(), m.name
    tf.extractall(out, filter='data')
smoke = out / 'frontend/smoke'
cfg = load(smoke / 'protocol.json')
scores = json.loads((smoke / 'scores.json').read_text())
stored = json.loads((smoke / 'assessment.json').read_text())
assert assess(cfg, scores) == stored
frozen = json.loads((smoke / 'frozen.json').read_text())
assert set(frozen) == {f'{a}_r{r}.tum' for a in cfg['arms'] for r in range(1,4)}
times = None
for name, sha in frozen.items():
    assert hashlib.sha256((smoke / name).read_bytes()).hexdigest() == sha
    rows = [list(map(float, ln.split())) for ln in (smoke / name).read_text().splitlines() if ln.strip()]
    assert all(len(row) == 8 and all(map(math.isfinite, row)) for row in rows)
    stamp = [row[0] for row in rows]
    assert all(b > a for a,b in zip(stamp, stamp[1:]))
    if times is None:
        times = stamp
    assert stamp == times
    assert len(rows) >= cfg['minimum_poses']

def quantile(values, p):
    values=sorted(values)
    i=(len(values)-1)*p
    lo=math.floor(i); hi=math.ceil(i)
    return values[lo] + (values[hi]-values[lo])*(i-lo)

def stats(values):
    return dict(n=len(values), mean_ms=statistics.mean(values), median_ms=statistics.median(values), p95_ms=quantile(values,.95))

profiles = {}
for run in scores['runs']:
    name = f"{run['arm']}_r{run['repetition']}"
    text = re.sub(r'\x1b\[[0-9;]*m', '', (smoke/f'{name}.log').read_text())
    values={}
    section=None
    for line in text.splitlines():
        if 'LIO Mapping Time' in line: section='LIO'
        elif 'VIO Time' in line: section='VIO'
        m=re.search(r'\|\s*(Current Total Time|ICP|computeJacobianAndUpdateEKF)\s*\|\s*([0-9.]+)',line)
        if m and section:
            values.setdefault(section+' '+m[1],[]).append(float(m[2])*1000)
    extraction=[float(v) for v in re.findall(r'extraction_ms=([0-9.eE+-]+)',text)]
    if extraction: values['unused_boundary_extraction']=extraction
    profiles[name]={k:stats(v) for k,v in values.items()}
(smoke/'stage-profiles.json').write_text(json.dumps(profiles, indent=2)+'\n')
verification=dict(archive_sha256=digest, archive_bytes=archive.stat().st_size, source_hashes=len(cfg['frozen_sources']),
                  trajectory_hashes=len(frozen), identical_timestamps=True, poses=len(times), assessment_reproduced=True,
                  status=stored['status'], timings_scope='CPU runner at 0.25x with cached labels; component timing, not an online GPU inference result')
(smoke/'independent-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
print('arm | mean ATE | mean 1s translation RPE | mean 1s rotation RPE | per-repeat LIO median ms')
for arm in cfg['arms']:
    row=stored['summary'][arm]
    print(arm, *(f'{row[m]["mean"]:.9f}' for m in ('ate_rmse_m','rpe_1s_translation_rmse_m','rpe_1s_rotation_rmse_deg')),
          [round(profiles[f'{arm}_r{r}']['LIO Current Total Time']['median_ms'],3) for r in range(1,4)])
print(json.dumps({k:stored[k] for k in ('mechanism','comparisons_to_geometry','increment_over_association')},indent=2))
