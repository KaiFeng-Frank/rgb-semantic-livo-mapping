"""Plot and archive the measured result, including negative outcomes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.loop_closure.common import digest, save, S


def render(out, comparison, build):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = ['ordinary', 'group_balanced', 'revisit_balanced']
    names = ['Ordinary fusion', 'Group-balanced', 'Revisit-balanced']
    metric = 'miou9_fixed_classes_abstain_wrong'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax, mode, scope, title in [(axes[0], 'all', 'full', 'All 1,101 scans / full map'),
                                    (axes[1], 'heldout', 'roi', 'Held-out scans / revisit ROI (primary)')]:
        values = [comparison['reports'][mode][arm][scope][metric] for arm in arms]
        ax.bar(names, values, color=['#718096', '#d69e2e', '#2b6cb0'])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Common-9 mIoU (%) / abstentions wrong')
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(i, value+1, f'{value:.3f}', ha='center', fontsize=10)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Same geometry, observations and scoring points; offline mapping', fontsize=12)
    fig.savefig(out / 'metrics.png', dpi=170)
    plt.close(fig)
    with np.load(out / 'heldout-maps.npz') as z:
        xyz = z['xyz']
        labels = {arm: z[arm] for arm in arms}
    center = np.asarray(build['roi_center'])
    selected = np.flatnonzero(np.square(xyz[:, :2]-center[:2]).sum(1) <= build['roi_radius_m']**2)
    # Fixed spatial subsampling is identical for both pictures, never chosen by GT.
    selected = selected[::max(1, (len(selected)+149999)//150000)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    for ax, arm, title in zip(axes[:2], ['ordinary', 'revisit_balanced'], ['Ordinary', 'Revisit-balanced']):
        ax.scatter(xyz[selected, 0], xyz[selected, 1], c=S.CLASS_COLORS[labels[arm][selected]]/255., s=.35, rasterized=True)
        ax.set_title(title)
    ax = axes[2]
    ax.scatter(xyz[selected, 0], xyz[selected, 1], c='#cbd5e0', s=.35, rasterized=True)
    changed = np.flatnonzero(labels['ordinary'] != labels['revisit_balanced'])
    ax.scatter(xyz[changed, 0], xyz[changed, 1], c='#e53e3e', s=2, rasterized=True)
    ax.set_title(f'Revised voxels: {len(changed):,} (red)')
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(center[0]-build['roi_radius_m'], center[0]+build['roi_radius_m'])
        ax.set_ylim(center[1]-build['roi_radius_m'], center[1]+build['roi_radius_m'])
        ax.set_xlabel('World x (m)')
        ax.set_ylabel('World y (m)')
    fig.suptitle('Held-out semantic-map comparison: coordinates are identical', fontsize=12)
    fig.savefig(out / 'map-comparison.png', dpi=170)
    plt.close(fig)


def package(out):
    out.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((out / 'comparison.json').read_text()) if (out / 'comparison.json').exists() else None
    build = json.loads((out / 'build.json').read_text()) if (out / 'build.json').exists() else None
    if comparison and build:
        render(out, comparison, build)
    save(out / 'execution.json', dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        run_url=f'https://github.com/{os.environ.get("GITHUB_REPOSITORY")}/actions/runs/{os.environ.get("GITHUB_RUN_ID")}',
        complete_comparison=comparison is not None,
        packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
        raw_data_included=False, training=False, official_leaderboard=False))
    body = '# Revisit semantic refinement at fixed geometry\n\n'
    body += 'Frozen B0 seed-1 predictions and the previously verified corrected seq07 trajectory. '
    body += 'GT is read only after all semantic maps are built. Offline future-aware common-9 mapping; not a single-scan or 19-class leaderboard result.\n\n'
    if comparison:
        body += '| Primary: held-out revisit ROI | mIoU9, abstentions wrong |\n|---|---:|\n'
        for arm, value in comparison['primary_percent'].items():
            body += f'| {arm} | {value:.6f}% |\n'
        body += f"\nRevisit minus ordinary: {comparison['delta_revisit_vs_ordinary_pp']:+.6f} percentage points. "
        body += f"Revisit minus group-balanced: {comparison['delta_revisit_vs_group_balanced_pp']:+.6f} percentage points.\n\n"
        body += 'This is a one-sequence feasibility result. Keep all arms, harmful changes and unchanged predictions in the comparison. '
        body += 'Do not interpret confidence-vote shares as calibrated probabilities or temporal groups as statistically independent.\n'
    else:
        body += '**Incomplete run:** consult the execution URL and available diagnostics. An archive is not evidence of success.\n'
    (out / 'README.md').write_text(body)
    assets = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(out.iterdir())
              if p.is_file() and p.name not in ['manifest.json', 'SHA256SUMS']]
    save(out / 'manifest.json', dict(assets=assets, raw_data_included=False))
    (out / 'SHA256SUMS').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in assets))


if __name__ == '__main__':
    package(Path(sys.argv[1]))
