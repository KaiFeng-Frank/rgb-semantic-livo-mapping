"""Reconcile saved training budgets, checkpoint selection, and point totals."""
import argparse
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import digest
from .repair_supervision_experiment import ARMS,STATIC,summarize


def validate(out):
    import torch
    torch.set_num_threads(1)
    protocol=read(out/'protocol.json');manifest=read(out/'feature-manifest.json')
    cfg=protocol['config']['optimization'];reports=[];checks=[]
    assert {e['part'] for e in manifest['files']}=={'train','dev'}
    expected_steps=sum((sum(e['part']=='train' and e['scene']==s and e['frame']>0 for e in manifest['files'])+3)//4 for s in protocol['parts']['train'])
    expected_queries=sum(manifest['train_groups'])
    static_reference=None
    for seed in cfg['seeds']:
        folder=out/'run'/f'seed-{seed}';report=read(folder/'evaluation.json');selection=read(folder/'selected.json')
        epochs=[read(folder/f'epoch-{i}.json') for i in range(1,cfg['epochs']+1)]
        for ep in epochs:
            for arm in ARMS:
                stats=ep['train'][arm]
                assert stats['frames']==626 and stats['steps']==expected_steps
                assert stats['queries']==expected_queries and stats['groups']==manifest['train_groups']
                assert np.isfinite(stats['loss']) and np.isfinite(stats['unreduced_loss_mass']).all()
        checkpoint_hashes={}
        for arm in ARMS:
            values=[ep['development']['arms'][arm]['miou_fixed16'] for ep in epochs]
            chosen=int(np.argmax(values))+1;sel=selection[arm];path=folder/(arm+'.pt')
            assert sel['epoch']==chosen and sel['dev_miou']==values[chosen-1]
            assert sel['sha256']==digest(path)
            assert report['arms'][arm]==epochs[chosen-1]['development']['arms'][arm]
            ck=torch.load(path,map_location='cpu',weights_only=True)
            assert sum(t.numel() for t in ck['fusion'].values())==15760
            assert ck['provenance']['deployable'] is False and ck['trained'] is True
            assert ck['provenance']['epoch']==chosen and ck['provenance']['seed']==seed
            assert ck['provenance']['anchor_sha256']==protocol['anchor_sha256']
            assert ck['provenance']['protocol_sha256']==digest(out/'training-protocol.json')
            assert ck['provenance']['feature_manifest_sha256']==digest(out/'feature-manifest.json')
            checkpoint_hashes[arm]=digest(path)
        frames=read(folder/'frames.json');assert len(frames)==321
        for arm,score in report['arms'].items():
            matrix=np.sum([np.asarray(s[arm]['confusion']) for s in report['per_scene'].values()],axis=0)
            np.testing.assert_array_equal(matrix,score['confusion'])
            union=matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
            miou=np.divide(matrix.diagonal(),union,out=np.zeros(16),where=union>0).mean()
            assert miou==score['miou_fixed16']
            fg=sum(f['camera_foreground_points'] for f in frames)
            correct=sum(f['camera_foreground_correct'][arm] for f in frames)
            assert fg==score['counts']['camera_foreground_points']
            assert correct==score['counts']['camera_foreground_correct']
            assert correct/fg==score['camera_foreground_accuracy']
        static={a:report['arms'][a] for a in STATIC}
        if static_reference is not None:assert static==static_reference
        static_reference=static;reports.append(report)
        checks.append(dict(seed=seed,frames_per_epoch=626,steps_per_epoch=expected_steps,
            queries_per_epoch=expected_queries,development_frames=321,checkpoint_hashes=checkpoint_hashes))
    assert summarize(reports)==read(out/'report.json')
    result=dict(passed=True,seeds=checks,cache_parity_frames=manifest['parity_frames'],
        real_output_checks=manifest['real_output_checks'],source='Saved epoch reports, selected checkpoints, per-scene confusion and per-frame foreground counts.',
        caveat='This checks internal result consistency, not new generalization or online delivery.')
    save(out/'validation.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    validate(p.parse_args().out)
