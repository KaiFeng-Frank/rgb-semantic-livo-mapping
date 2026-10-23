"""Post-primary attribution controls: remove one visit's correction evidence.

No tuning or rebuilding of the three primary maps. The original fallback map
and geometric/protection masks remain identical, so 'visit only' describes the
correction's votes, not a claim that the entire map has only that visit's input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tools.loop_closure.common import digest, save
from .fusion import apply_proposal
from .run import configuration, score, verify_configuration

SOURCE_RELEASE = 'loop-semantics-40036755735-1'
SOURCE_MANIFEST_SHA256 = '418f036474ec8b365b777fc58c516a975514a581e340d7e9bdd0129888aa73e1'
ARMS = ['first_visit_only', 'second_visit_only']


def one_visit_proposal(evidence, which, cfg):
    values = evidence['score_' + which]
    ordered = np.sort(values, axis=1)
    total = values.sum(1)
    share = np.divide(ordered[:, -1], total, out=np.zeros(len(values)), where=total > 0)
    margin = np.divide(ordered[:, -1]-ordered[:, -2], total, out=np.zeros(len(values)), where=total > 0)
    arm = which + '_visit_only'
    return dict(keys=evidence['keys'], **{arm: dict(label=values.argmax(1).astype(np.uint8),
         accept=evidence['eligible'] & (share >= cfg['minimum_winner_share']) & (margin >= cfg['minimum_winner_margin']))})


def build_controls(source, out):
    cfg = configuration()
    if digest(source / 'manifest.json') != SOURCE_MANIFEST_SHA256:
        raise ValueError('Attribution source manifest changed')
    manifest = json.loads((source / 'manifest.json').read_text())
    assets = {r['name']: r for r in manifest['assets']}
    required = ['build.json', 'comparison.json', 'protocol.json'] + [mode + suffix for mode in ['all', 'heldout']
                  for suffix in ['-maps.npz', '-revisit-evidence.npz']]
    for name in required:
        if digest(source / name) != assets[name]['sha256']:
            raise ValueError('Attribution input checksum mismatch: ' + name)
    verify_configuration(source / 'protocol.json')
    build = json.loads((source / 'build.json').read_text())
    evidence_report = dict(source_release=SOURCE_RELEASE, source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        introduced_after_primary_result=True, parameter_tuning=False, original_maps_rebuilt=False,
        description='Only the correction evidence is restricted to one visit. Original map fallback and all masks are shared.',
        modes={})
    out.mkdir(parents=True, exist_ok=True)
    for mode in ['all', 'heldout']:
        with np.load(source / f'{mode}-maps.npz') as z:
            mapping = {k: z[k] for k in z.files}
        with np.load(source / f'{mode}-revisit-evidence.npz') as z:
            evidence = {k: z[k] for k in z.files}
        original_hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in mapping.items()}
        for which in ['first', 'second']:
            arm = which + '_visit_only'
            proposal = one_visit_proposal(evidence, which, cfg)
            mapping[arm] = apply_proposal(mapping['key'], mapping['ordinary'], mapping['protected'], proposal, arm)
        path = out / f'{mode}-maps.npz'
        np.savez_compressed(path, **mapping)
        with np.load(path) as z:
            for k, checksum in original_hashes.items():
                if hashlib.sha256(z[k].tobytes()).hexdigest() != checksum:
                    raise AssertionError('An original map or geometry array changed')
        build['modes'][mode]['map_sha256'] = digest(path)
        evidence_report['modes'][mode] = dict(original_arrays_unchanged=True, array_hashes=original_hashes,
            changed_voxels={arm: int(np.count_nonzero(mapping[arm] != mapping['ordinary'])) for arm in ARMS})
    save(out / 'build.json', build)
    shutil.copyfile(source / 'protocol.json', out / 'protocol.json')
    save(out / 'attribution-protocol.json', evidence_report)


def verify_original_scores(source, out):
    before = json.loads((source / 'comparison.json').read_text())
    after = json.loads((out / 'comparison.json').read_text())
    for mode in before['reports']:
        for arm in before['reports'][mode]:
            if before['reports'][mode][arm] != after['reports'][mode][arm]:
                raise AssertionError('Original scores changed in attribution run')
    after['attribution'] = json.loads((out / 'attribution-protocol.json').read_text())
    after['attribution']['original_scores_identical'] = True
    after['attribution']['group_minus_first_visit_pp'] = after['primary_percent']['group_balanced'] - after['primary_percent']['first_visit_only']
    after['attribution']['group_minus_second_visit_pp'] = after['primary_percent']['group_balanced'] - after['primary_percent']['second_visit_only']
    save(out / 'comparison.json', after)


def render_attribution(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison = json.loads((out / 'comparison.json').read_text())
    arms = ['ordinary', 'first_visit_only', 'second_visit_only', 'group_balanced', 'revisit_balanced']
    names = ['Ordinary fusion', 'Correction: first visit only', 'Correction: second visit only',
             'Correction: both visits / group balance', 'Correction: both visits / visit balance']
    values = [comparison['primary_percent'][arm] for arm in arms]
    fig, ax = plt.subplots(figsize=(10, 4.2), layout='constrained')
    ax.scatter(values, np.arange(len(arms)), s=70, c=['#718096', '#805ad5', '#319795', '#d69e2e', '#2b6cb0'])
    ax.axvline(values[0], color='#718096', ls='--', lw=1)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.4f}%', (value, i), xytext=(7, 7), textcoords='offset points')
    ax.set_yticks(range(len(arms)), names)
    ax.invert_yaxis()
    spread = max(max(values)-min(values), .2)
    ax.set_xlim(min(values)-.15*spread, max(values)+.45*spread)
    ax.set_xlabel('Held-out revisit ROI: common-9 mIoU (%), abstentions wrong')
    ax.set_title('Post-primary attribution; geometry, fallback and protection masks fixed')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(out / 'attribution.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    build_controls(a.source, a.out)
    score(a.root, a.out, additional_arms=ARMS)
    verify_original_scores(a.source, a.out)
    render_attribution(a.out)
