"""Real GPU feature capture + two-phase SHM checks on one existing scan.

This is a transport/runtime check, not a semantic accuracy experiment.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from .kalman_cache import gpu_admission
from tools.semantic_memory.nuscenes_pilot_data import DATA


def run(index,out):
    admission=gpu_admission()
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    from ptv3_client import PTv3Client
    from instance_features import pool_features
    torch.set_num_threads(1)
    row=next(iter(json.loads(index.read_text())['sequences'].values()))[0]
    points=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:4].copy();points[:,3]/=255.
    groups=np.zeros(len(points),np.uint16);groups[:40]=1;groups[40:100]=2
    seg=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    counter=[0]
    hook=seg.model.register_forward_hook(lambda *args:counter.__setitem__(0,counter[0]+1))
    seg.retain_instance_features=True
    labels,confidence,probabilities=seg.segment(points,return_probabilities=True)
    start=time.perf_counter()
    expected=pool_features(*seg._retained_features,groups,2).cpu().numpy()
    pool_ms=(time.perf_counter()-start)*1000
    assert counter[0]==1
    hook.remove();del seg;torch.cuda.empty_cache()
    out.mkdir(parents=True,exist_ok=False)
    client=PTv3Client(half=True,shuffle=False,intensity_scale=1.,probability_classes=16,
        instance_features=True,stderr_path=str(out/'worker.log'))
    times=[];cross_forward=[]
    try:
        for i in range(6):
            lab,conf,prob=client.segment_probabilities(points)
            before=prob.copy()
            pooled=client.pool_last_instances(groups,2)
            again=client.pool_last_instances(groups,2)
            np.testing.assert_allclose(pooled,again,atol=1e-5,rtol=1e-5)
            np.testing.assert_array_equal(prob,before)
            # Separate forwards/processes can differ in spconv rounding. Record,
            # don't mistake that for a same-forward transport parity violation.
            cross_forward.append(dict(label_changes=int((lab!=labels).sum()),
                pooled_max_abs=float(np.abs(pooled-expected).max())))
            assert client.pool_last_instances(np.zeros(len(points),np.uint16),0).shape==(0,128)
            times.append(client.last_infer_ms)
        f_lab,f_conf,f_prob,f_pool=client.segment_instances(points,groups,2)
        np.testing.assert_allclose(f_pool,client.pool_last_instances(groups,2),atol=1e-5,rtol=1e-5)
    finally:client.close()
    report=dict(admission=admission,scope='one real scan, engineering only',points=len(points),
        same_forward_count=counter[0],retained_and_eager_pool_agree=True,raw_output_unchanged=True,
        ring_wraparound=True,empty_groups=True,pool_ms=pool_ms,worker_ms=times,
        independent_forward_diagnostic=cross_forward,
        pooled_bytes=expected.nbytes,all_point_feature_copy=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.index,a.out)
