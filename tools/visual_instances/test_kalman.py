"""Causality, revisability, feature transport and neural export contracts."""
import json
import numpy as np
import pytest
import torch

from .kalman import EvidenceFilter, ObservationModel, INPUT_DIM, VAR_MIN, decision
from .train_kalman import Head,train_scene
from .semantics import Config
from .geometry import InstanceFrame,mask_frame
from src.instance_features import pool_features


def test_reliable_history_survives_bad_observation_but_can_be_overturned():
    f=EvidenceFilter();feat=np.ones(64,np.float32)
    car=np.zeros(16);car[3]=6
    truck=np.zeros(16);truck[9]=6
    for i in range(30):h,s=f.step((1,1),car,np.ones(16)*.2,feat,1+i*500_000_000)
    h,s=f.step((1,1),truck,np.ones(16)*20,feat,15_000_000_001)
    assert h.argmax()==3
    for i in range(5):h,s=f.step((1,1),truck,np.ones(16)*.2,feat,15_500_000_001+i*500_000_000)
    assert h.argmax()==9
    assert np.min(f.states[(1,1)]['variance'])>=VAR_MIN


def test_no_cross_epoch_ttl_or_reordered_inheritance():
    f=EvidenceFilter();z=np.arange(16,dtype=float);r=np.ones(16);feat=np.ones(64)
    f.step(('a',1,1),z,r,feat,1)
    h,s=f.step(('a',1,2),-z,r,feat,2)
    np.testing.assert_array_equal(h,-z);assert not s['history']
    with pytest.raises(ValueError):f.step(('a',1,2),z,r,feat,2)
    h,s=f.step(('a',1,2),z,r,feat,3_000_000_000)
    np.testing.assert_array_equal(h,z);assert not s['history']


def test_raw_observations_not_filtered_outputs_are_retained():
    f=EvidenceFilter();r=np.ones(16);feat=np.ones(64)
    f.step(1,np.arange(16),r,feat,1)
    raw=np.arange(16)[::-1].copy()
    h,_=f.step(1,raw,r,feat,500_000_001)
    assert not np.array_equal(h,raw)
    np.testing.assert_array_equal(f.states[1]['history'][-1][0],raw)


def test_numpy_torch_export_equivalence_and_gradients(tmp_path):
    torch.manual_seed(1)
    model=Head(np.zeros(INPUT_DIM),np.ones(INPUT_DIM))
    with torch.no_grad():
        model.delta.weight.normal_(0,.03);model.variance.weight.normal_(0,.02)
    path=tmp_path/'head.json';model.export(path,{})
    other=ObservationModel(path)
    x=torch.randn(4,INPUT_DIM);raw=torch.rand(4,16).softmax(-1)
    z,r=model(x,raw);nz,nr=other(x.numpy(),raw.numpy())
    np.testing.assert_allclose(nz,z.detach().numpy(),atol=2e-6)
    np.testing.assert_allclose(nr,r.detach().numpy(),atol=2e-6)
    observations=[]
    for i in range(4):
        hist=np.zeros(16);hist[3]=12
        observations.append(dict(track=1,epoch=1,stamp=i*500_000_000+1,x=x[i].numpy(),raw_mean=raw[i].numpy(),targets=dict(histogram=hist)))
    loss=train_scene(model,dict(observations=observations),True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.variance.weight.grad.abs().sum()>0


def test_point_inverse_pooling_and_empty_groups():
    features=torch.arange(4*64,dtype=torch.float32).reshape(4,64)
    inverse=torch.tensor([3,0,1,3,2])
    groups=np.array([1,1,0,2,2],np.uint16)
    result=pool_features(features,inverse,groups,2)
    expanded=features[inverse]
    for i in range(2):
        torch.testing.assert_close(result[i,:64],expanded[groups==i+1].mean(0))
        torch.testing.assert_close(result[i,64:],expanded[groups==i+1].std(0,unbiased=False))
    assert pool_features(features,inverse,np.zeros(5,np.uint16),0).shape==(0,128)


def test_crowded_frame_excess_instances_stay_unpooled():
    from .semantics import instance_groups
    ids=np.repeat(np.arange(1,131),6)
    xyz=np.zeros((len(ids),3),np.float32)
    association=dict(ids=ids,labels=np.full(len(ids),3),info=dict(session='a'*32))
    groups,tracks=instance_groups(xyz,association)
    assert len(tracks)==128 and int(groups.max())==128
    assert not groups[ids>128].any()
    groups,tracks=instance_groups(xyz,association,Config(max_tracks=2))
    assert tracks==[1,2] and not groups[ids>2].any()


def test_visual_scores_survive_packet_and_mask_filter():
    masks=np.ones((3,10,10),bool);evidence=np.arange(240).reshape(3,80)/240.
    frame=mask_frame(masks,[8,9,10],[2,2,7],[.9,.2,.8],1,'a'*32,(10,10),evidence)
    decoded=InstanceFrame.decode(frame.encode())
    np.testing.assert_array_equal(decoded.ids,[8,10])
    np.testing.assert_allclose(decoded.class_scores,evidence[[0,2]])
    pytest.importorskip('ultralytics')
    from .evidence_predictor import EvidenceResults
    r=EvidenceResults(np.zeros((10,10,3),np.uint8),'test',{0:'x'},
        boxes=torch.zeros((3,6)),class_scores=torch.tensor(evidence))
    np.testing.assert_allclose(r[[2,0]].cpu().numpy().class_scores,evidence[[2,0]])


def test_history_cannot_override_competing_truck_but_background_can_be_repaired():
    h=np.zeros(16);h[3]=6;now=np.zeros(16);now[14]=2
    raw=np.zeros((6,16));raw[:,14]=1
    assert decision(h,now,raw,Config())[2]
    raw[0]=0;raw[0,9]=.7;raw[0,3]=.3
    assert not decision(h,now,raw,Config())[2]


def test_retained_feature_transport_has_owned_slots_and_preserves_raw_tuple(tmp_path,monkeypatch):
    import sys
    from src import ptv3_client as ipc
    from tools.semantic_memory.test_probability_transport import WORKER
    worker=WORKER.replace("os.write(1,b'RDY0'", """groups=np.ndarray((s,n),dtype=np.uint16,buffer=shm.buf,offset=s*n*(22+c*4))
pooled=np.ndarray((s,128,128),dtype=np.float32,buffer=shm.buf,offset=s*n*(24+c*4))
last=None
os.write(1,b'RDY0'""")
    worker=worker.replace("assert tag == b'H'", "assert tag in (b'H',b'G',b'F')")
    worker=worker.replace("    lab[slot,:count] = seq%c", """    if tag in (b'F',b'G'):
        k=struct.unpack('<I',sys.stdin.buffer.read(4))[0]
        if tag==b'G': assert last==(seq,slot,count)
        pooled[slot,:k]=seq+np.arange(k)[:,None]+groups[slot,:count].sum()
    last=(seq,slot,count)
    lab[slot,:count] = seq%c""")
    worker=worker.replace('lab=conf=prob=None','lab=conf=prob=groups=pooled=None')
    path=tmp_path/'worker.py';path.write_text(worker)
    monkeypatch.setattr(ipc,'PTV3_PY',sys.executable);monkeypatch.setattr(ipc,'WORKER',str(path))
    client=ipc.PTv3Client(probability_classes=16,instance_features=True,max_pts=8)
    try:
        saved=[]
        for i in range(10):
            raw=client.segment_probabilities(np.zeros((4,4),np.float32))
            assert len(raw)==3
            value=client.pool_last_instances(np.array([0,1,1,2],np.uint16),2)
            np.testing.assert_array_equal(value[:,0],[i+4,i+5]);saved.append(value)
        for i,value in enumerate(saved):np.testing.assert_array_equal(value[:,0],[i+4,i+5])
        assert client.pool_last_instances(np.zeros(4,np.uint16),0).shape==(0,128)
        with pytest.raises(ValueError):client.pool_last_instances(np.array([0,1,2,3]),2)
        client.submit(np.zeros((4,4),np.float32))
        with pytest.raises(RuntimeError):client.pool_last_instances(np.zeros(4,np.uint16),0)
        client.collect()
        assert len(client.segment(np.zeros((4,4),np.float32)))==2
    finally:client.close()
    assert client.p.returncode==0
