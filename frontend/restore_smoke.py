"""Recover only a frozen short KITTI segment on an ephemeral validation runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools'))
from restore_driving_comparison import RemoteZip, client, download
from compare_driving_semantics import RELEASE
BASE='https://s3.eu-central-1.amazonaws.com/avg-kitti/'
DRIVE='2011_09_30_drive_0027'


def save_member(z, member, dest, records):
    b=z.read(member)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    records.append(dict(member=member,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))


def restore(root):
    cfg=json.loads((REPO/'frontend/protocol.json').read_text())['short_sequence']
    first,end=cfg['first_frame'],cfg['end_exclusive']
    if first!=0 or end>120:raise ValueError('only the admitted 120-frame startup smoke is authorized here')
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    raw=root/'raw/2011_09_30'/f'{DRIVE}_sync'
    session=client()
    for name in ['2011_09_30_calib.zip']:
        response=session.get(BASE+'raw_data/'+name,timeout=60);response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for member in z.namelist():
                if member.endswith('.txt'):save_member(z,member,root/'raw'/member,records)
    url=BASE+f'raw_data/{DRIVE}/{DRIVE}_sync.zip'
    def shard(worker):
        r=RemoteZip(url);found=[]
        with zipfile.ZipFile(r) as z:
            names=[f'{folder}/data/{f:010d}.{ext}' for f in range(first+worker,end,4)
                   for folder,ext in [('velodyne_points','bin'),('image_02','png')]]
            if worker==0:names+=['velodyne_points/timestamps_start.txt','velodyne_points/timestamps_end.txt','image_02/timestamps.txt']
            for rel in names:save_member(z,f'2011_09_30/{DRIVE}_sync/{rel}',raw/rel,found)
        return found,r.fetched
    with ThreadPoolExecutor(max_workers=4) as pool:
        shards=list(pool.map(shard,range(4)))
    for rows,_ in shards:records+=rows
    print('SHORT_RAW_RECOVERED',end-first,'frames',sum(n for _,n in shards),'range bytes',flush=True)
    # True 100 Hz IMU from extract archive. Trim/reindex only the selected time window;
    # never fall back to the synchronized archive's 10 Hz OXTS.
    low=np.asarray((raw/'velodyne_points/timestamps_start.txt').read_text().splitlines(),dtype='datetime64[ns]')[first].astype(np.int64)-200_000_000
    high=np.asarray((raw/'velodyne_points/timestamps_end.txt').read_text().splitlines(),dtype='datetime64[ns]')[end-1].astype(np.int64)+200_000_000
    r=RemoteZip(BASE+f'raw_data/{DRIVE}/{DRIVE}_extract.zip')
    with zipfile.ZipFile(r) as z:
        prefix=f'2011_09_30/{DRIVE}_extract/oxts/'
        lines=z.read(prefix+'timestamps.txt').decode().splitlines()
        stamps=np.asarray(lines,dtype='datetime64[ns]').astype(np.int64)
        indices=np.flatnonzero((stamps>=low)&(stamps<=high))
        imu=root/'oxts100';(imu/'data').mkdir(parents=True,exist_ok=True)
        (imu/'timestamps.txt').write_text('\n'.join(lines[i] for i in indices)+'\n')
        for j,i in enumerate(indices):save_member(z,prefix+f'data/{i:010d}.txt',imu/f'data/{j:010d}.txt',records)
    print('IMU_RECOVERED',len(indices),flush=True)
    assets={a['name']:a for a in json.loads((REPO/'results/v06/artifacts_20260928/artifact-manifest.json').read_text())['assets']}
    for arm in ['armB0_s1','ZS']:
        name=f'v06-pred-{arm}-seq07-r1.tar'
        archive=root/name
        download(session,RELEASE+name,archive,assets[name]['sha256'])
        index_name=name+'.index.json';download(session,RELEASE+index_name,root/index_name,assets[index_name]['sha256'])
        index={r['path']:r for r in json.loads((root/index_name).read_text())}
        with tarfile.open(archive) as tf:
            for f in range(first,end):
                rel=f'out/v06_score/pred/{arm}_07_r1/f{f:06d}.npz';data=tf.extractfile(rel).read()
                if hashlib.sha256(data).hexdigest()!=index[rel]['sha256']:raise ValueError('prediction hash mismatch')
                p=root/f'pred/{arm}/f{f:06d}.npz';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        print('PREDICTIONS_VERIFIED',arm,end-first,flush=True)
    evidence=dict(protocol=cfg,records=records,raw_uploaded=False,new_inference=False,
                  interpretation='cached-label mechanism test, not an online latency result',
                  raw_range_bytes=sum(n for _,n in shards),imu_range_bytes=r.fetched)
    (root/'input-recovery.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    restore(ap.parse_args().root)
