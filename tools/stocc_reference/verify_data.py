"""Audit the retained validation files against saved manifests and nuScenes tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path('/home/wuyou/stocc_reference')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels-only', action='store_true')
    parser.add_argument('--cameras-only', action='store_true', help='Reuse the completed label audit; verify newly completed cameras.')
    args=parser.parse_args()
    assert not (args.labels_only and args.cameras_only)
    data=ROOT/'data/nuscenes'
    plan=json.loads((ROOT/'camera-plan.json').read_text())
    scenes={s['token']:s['name'] for s in json.loads((data/'v1.0-trainval/scene.json').read_text()) if s['name'] in plan['scenes']}
    tokens={s['token']:scenes[s['scene_token']] for s in json.loads((data/'v1.0-trainval/sample.json').read_text()) if s['scene_token'] in scenes}
    assert len(tokens)==6019 and len(scenes)==150
    labels=json.loads((ROOT/'occ3d-label-manifest.json').read_text())
    assert len(labels)==len(tokens)
    found=set();classes=set();start=time.time()
    for row in ([] if args.cameras_only else labels):
        path=data/row['path'];scene,token=path.parts[-3:-1]
        assert token not in found and tokens[token]==scene
        found.add(token)
        content=path.read_bytes()
        assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
        with np.load(path) as label:
            assert set(label.files)=={'semantics','mask_lidar','mask_camera'}
            for key in label.files: assert label[key].shape==(200,200,16)
            semantics=label['semantics'];classes.update(np.unique(semantics).tolist())
            assert semantics.min()>=0 and semantics.max()<=17
            for key in ('mask_camera','mask_lidar'):
                values=label[key];assert values.min()>=0 and values.max()<=1
    if args.cameras_only:
        audit=(ROOT/'label-audit.json').read_bytes()
        result=json.loads(audit)
        assert result['labels']==6019 and result['label_sha256_and_shape_verified']
        result['reused_label_audit_sha256']=hashlib.sha256(audit).hexdigest()
    else:
        assert found==set(tokens)
        result=dict(labels=len(found),scenes=len(scenes),classes=sorted(classes),
                    label_sha256_and_shape_verified=True,elapsed_seconds=time.time()-start)
    if not args.labels_only:
        complete=json.loads((ROOT/'camera-acquisition/complete.json').read_text())
        assert complete['complete'] and complete['images']==36114
        camera_records=[r for index in range(1,11) for r in json.loads((ROOT/f'camera-acquisition/{index:02d}/files.json').read_text())]
        assert len(camera_records)==len(plan['required'])==36114
        required=set(plan['required']);found=set();dimensions=set()
        for row in camera_records:
            assert row['path'] in required and row['path'] not in found
            found.add(row['path']);path=data/row['path'];content=path.read_bytes()
            assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
            with Image.open(path) as img:
                assert img.format=='JPEG';dimensions.add(img.size);img.verify()
        assert found==required
        result.update(cameras=len(found),camera_sha256_and_jpeg_verified=True,image_dimensions=sorted(dimensions))
    result.update(completed=time.time(),elapsed_seconds=time.time()-start)
    name='label-audit.json' if args.labels_only else 'data-audit.json'
    (ROOT/name).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
