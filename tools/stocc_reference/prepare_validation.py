"""Use official transforms on validation only, reusing one SDK instance.

The official entry point also converts 28,130 unused training frames and reloads
the large SDK three times. This wrapper changes only sample selection and SDK
reuse. All camera/pose/box/global-info calculations remain upstream functions.
"""
import json
import os
from pathlib import Path
import pickle
import time

from nuscenes import NuScenes
from tools import create_data_bevdet as prep

ROOT=Path('/home/wuyou/stocc_reference')


class SelectedSamples:
    def __init__(self, sdk, samples):
        self.sdk=sdk
        self.sample=samples

    def __getattr__(self,name):
        # SDK methods stay bound to the full original tables and token indices.
        return getattr(self.sdk,name)


def main():
    os.chdir(ROOT/'upstream')
    plan=json.loads((ROOT/'camera-plan.json').read_text())
    sdk=NuScenes('v1.0-trainval', './data/nuscenes/',verbose=True)
    scenes={s['token']:s['name'] for s in sdk.scene if s['name'] in plan['scenes']}
    samples=[s for s in sdk.sample if s['scene_token'] in scenes]
    assert len(samples)==6019 and len(scenes)==150
    prep.NuScenes=lambda *args,**kwargs:sdk

    def write_infos(tag, selected):
        started=time.time()
        _, infos=prep.nuscenes_converter._fill_trainval_infos(
            SelectedSamples(sdk,selected),set(),set(scenes),test=False,max_sweeps=0)
        for split, rows in [('train',[]),('val',infos)]:
            with open(f'./data/nuscenes/{tag}_infos_{split}.pkl','wb') as stream:
                pickle.dump(dict(infos=rows,metadata=dict(version='v1.0-trainval')),stream)
        prep.add_ann_adj_info(tag)
        prep.add_global_info(tag)
        print('metadata_complete',tag,len(infos),time.time()-started,flush=True)
        return infos

    available=[]
    for token, name in sorted(scenes.items(),key=lambda pair:pair[1]):
        selected=[s for s in samples if s['scene_token']==token]
        ready=all(Path(sdk.get_sample_data_path(s['data'][cam])).is_file()
                  for s in selected for cam in s['data'] if cam.startswith('CAM_'))
        if ready:
            available.append((name,selected))
            break
    assert available, 'No complete scene cameras available yet'
    name, selected=available[0]
    write_infos('stocc-smoke-nuscenes',selected)
    (ROOT/'smoke-scene.json').write_text(json.dumps(dict(scene=name,frames=len(selected),
        ann_file='data/nuscenes/stocc-smoke-nuscenes_infos_global_val.pkl'),indent=2))
    write_infos('bevdetv4-nuscenes',samples)
    (ROOT/'metadata-complete.json').write_text(json.dumps(dict(val_frames=6019,val_scenes=150,
        smoke_scene=name,source_functions_unmodified=True,completed=time.time()),indent=2))


if __name__=='__main__':main()
