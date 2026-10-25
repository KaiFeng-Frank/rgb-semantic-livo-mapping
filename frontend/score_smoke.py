"""Read official pose GT only after all frontend trajectories have been frozen."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tools.loop_closure.common import timestamps,matrices,interpolate,check_time_tables,S
from frontend.protocol import load, raw_dir


def score(root,out):
    cfg=load(out/'protocol.json')
    if cfg.get('diagnostic'):
        raise ValueError('diagnostic development replay cannot be used for accuracy scoring')
    frozen=json.loads((out/'frozen.json').read_text())
    runs=json.loads((out/'runs.json').read_text())
    expected={f'{arm}_r{rep+1}.tum' for arm in cfg['arms'] for rep in range(cfg['repetitions'])}
    if set(frozen)!=expected or len(runs)!=len(expected):raise ValueError('incomplete paired runs')
    for name,digest in frozen.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:raise ValueError('trajectory changed before scoring')
    base='https://s3.eu-central-1.amazonaws.com/avg-kitti/'
    gt_files={}
    seq=cfg['short_sequence']['sequence']
    for archive,members in [('data_odometry_calib.zip',[f'dataset/sequences/{seq}/calib.txt',f'dataset/sequences/{seq}/times.txt']),('data_odometry_poses.zip',[f'dataset/poses/{seq}.txt'])]:
        data=urllib.request.urlopen(base+archive,timeout=90).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in members:gt_files[Path(member).name]=z.read(member)
    tr=np.eye(4)
    for line in gt_files['calib.txt'].decode().splitlines():
        if line.startswith('Tr:'):tr[:3]=np.array(list(map(float,line.split()[1:]))).reshape(3,4)
    poses=np.loadtxt(io.BytesIO(gt_files[seq+'.txt'])).reshape(-1,3,4)
    gt=np.tile(np.eye(4),(len(poses),1,1));gt[:,:3]=poses;gt=np.linalg.inv(tr)@gt@tr
    # The raw drive has 1106 images; official odometry seq07 covers the first 1101.
    # Keep the same coverage convention as the existing full-sequence evaluator.
    ts=timestamps(raw_dir(root,cfg)/'image_02/timestamps.txt')[:len(gt)]/1e9
    clock=check_time_tables(np.loadtxt(io.BytesIO(gt_files['times.txt'])),ts)
    scores=[]
    for run in runs:
        name=f"{run['arm']}_r{run['repetition']}.tum"
        e=np.loadtxt(out/name);pred=matrices(e)@S.T_I_L;truth=interpolate(ts,gt,e[:,0])
        source,target=pred[:,:3,3],truth[:,:3,3]
        a,b=source.mean(0),target.mean(0)
        u,_,vt=np.linalg.svd((source-a).T@(target-b)/len(source));d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
        rot=vt.T@d@u.T;aligned=(source-a)@rot.T+b
        error=np.linalg.norm(aligned-target,axis=1)
        future=e[:,0]+1.;valid=future<=e[-1,0]
        de=np.linalg.inv(pred[valid])@interpolate(e[:,0],pred,future[valid])
        dg=np.linalg.inv(truth[valid])@interpolate(ts,gt,future[valid])
        delta=np.linalg.inv(dg)@de
        rpe=np.linalg.norm(delta[:,:3,3],axis=1)
        angles=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
        scores.append(dict(**run,ate_rmse_m=float(np.sqrt(np.mean(error**2))),
                           rpe_1s_translation_rmse_m=float(np.sqrt(np.mean(rpe**2))),
                           rpe_1s_rotation_rmse_deg=float(np.sqrt(np.mean(angles**2)))))
    scope=f"{cfg['short_sequence']} cached-label frontend validation; not full-sequence accuracy or real-time inference"
    result=dict(scope=scope,
                time_alignment=clock,runs=scores,gt_sha256={k:hashlib.sha256(v).hexdigest() for k,v in gt_files.items()})
    (out/'scores.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Semantic frontend short-sequence admission','',scope,'',
           '| Arm | Repetition | Poses | ATE m | RPE 1s m | RPE 1s deg | Updates with boundary constraints |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in scores:lines.append(f"| {r['arm']} | {r['repetition']} | {r['poses']} | {r['ate_rmse_m']:.6f} | {r['rpe_1s_translation_rmse_m']:.6f} | {r['rpe_1s_rotation_rmse_deg']:.6f} | {r['updates_with_constraints']} |")
    lines+=['','These data do not establish full-sequence or online superiority. No large experiment is authorized by this smoke alone.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();score(a.root,a.out)
