"""Preserve all successful/failed arms, trajectories and derived registration input."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.loop_closure.common import digest, save


def package(out):
    out.mkdir(parents=True, exist_ok=True)
    save(out/'execution.json', dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        run_url=f'https://github.com/{os.environ.get("GITHUB_REPOSITORY")}/actions/runs/{os.environ.get("GITHUB_RUN_ID")}',
        packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()))
    body = ['# Semantic-assisted pose refinement: seq07\n',
        'Fixed geometry, initial trajectory, accepted loop, graph and predicted labels. '
        'One known sequence / one loop; offline, no new training. G/D/S/T differ only in registration weights. '
        'D: movable-class downweighting; S: plus semantic compatibility; T: plus temporal reliability. '
        'ZS/B0 are archived pass-1 predictions. Lower is better for all table errors.\n']
    if (out/'comparison.json').exists():
        report = json.loads((out/'comparison.json').read_text())
        body += ['| Arm | ATE m | RPE 1s m | RPE 1s deg | Loop m | Loop deg | Fallback |',
                 '|---|---:|---:|---:|---:|---:|---|']
        for name, v in report['arms'].items():
            body.append(f"| {name} | {v['ate_rmse_m']:.6f} | {v['rpe_1s_translation_rmse_m']:.6f} | "
                        f"{v['rpe_1s_rotation_rmse_deg']:.6f} | {v['loop_error']['translation_m']:.6f} | "
                        f"{v['loop_error']['rotation_deg']:.6f} | {v['fallback']} |")
        body += ['\n![Pose comparison](pose-comparison.png)', '\n![Position error](error-over-time.png)',
                 '\nAll candidate trajectories, including any fallback, are scored separately in comparison.json. '
                 'Do not interpret the best row as a validated deployment choice. Improvements in one error may trade against another.']
    else:
        body.append('Incomplete run: no final comparison exists. Inspect execution and workflow logs; do not claim success.')
    body.append('\nregistration-inputs.npz contains derived voxel centroids, labels/confidence and temporal evidence for reproduction, '
                'not raw scans or full prediction archives. measurements.json records every candidate, guard and graph solve. '
                'Protocol files and SHA256SUMS freeze the experiment. Ground truth is confined to score.py.')
    (out/'README.md').write_text('\n'.join(body)+'\n')
    assets = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(out.iterdir())
              if p.is_file() and p.name not in ('manifest.json', 'SHA256SUMS')]
    save(out/'manifest.json', dict(assets=assets, raw_data_included=False, full_prediction_caches_included=False))
    (out/'SHA256SUMS').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in assets))


if __name__ == '__main__':
    package(Path(sys.argv[1]))
