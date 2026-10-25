"""Safety/causality tests; semantic effect is assessed on real held-out scans."""
from dataclasses import replace

import numpy as np
import pytest

from src.semantic_consistency import CausalConsistency, ConsistencyConfig, associate, fuse


XYZ = np.array([[5., 0, 0], [5., 1, 0], [5., 2, 0]])
POSE = np.eye(4)


def probs(label, confidence, count=3):
    p = np.full((count, 16), (1-confidence)/15, np.float32)
    p[:, label] = confidence
    return p


def run(core, points, p, stamp, pose=POSE):
    return core.step(points, p, stamp_ns=stamp, pose=pose, diagnostics=True)


def test_first_frame_exact_raw_then_soft_repair_without_recursive_writeback():
    core = CausalConsistency()
    history = probs(10, .99)
    current = probs(12, .4)
    current[:, 10] = .38
    current[:, [i for i in range(16) if i not in (10, 12)]] = .22/14
    assert np.array_equal(run(core, XYZ, history, 0)['probabilities'], history)
    r = run(core, XYZ, current, 500000000)
    assert r['status']['changed'] == 3
    assert np.all(r['probabilities'].argmax(1) == 10)
    assert np.array_equal(core.history['probabilities'], current)
    # A third weak frame must NOT see the previously enhanced posterior.
    r = run(core, XYZ, current, 1000000000)
    assert r['status']['used'] == 0
    assert np.array_equal(r['probabilities'], current)


@pytest.mark.parametrize('label,conf', [(12,.95), (6,.5), (3,.5)])
def test_current_evidence_veto_and_moving_classes_preserve_raw(label, conf):
    core = CausalConsistency()
    run(core, XYZ, probs(10,.99), 0)
    p = probs(label,conf)
    assert np.array_equal(run(core, XYZ, p, 500000000)['probabilities'], p)


def test_moving_history_not_used_to_relabel_static_surface():
    core = CausalConsistency()
    run(core, XYZ, probs(6,.99), 0)
    p = probs(12,.4)
    assert np.array_equal(run(core, XYZ, p, 500000000)['probabilities'], p)


def test_age_missing_geometry_and_capacity_fall_back_exactly():
    p = probs(12,.4)
    for xyz, stamp, cfg, reason in ((XYZ,2000000000,ConsistencyConfig(),'expired'),
                                   (XYZ+10,500000000,ConsistencyConfig(),'matched'),
                                   (np.tile(XYZ,(2,1)),500000000,replace(ConsistencyConfig(),max_points=3),'point_budget')):
        core = CausalConsistency(cfg)
        run(core,XYZ,probs(10,.99),0)
        cur = np.tile(p,(len(xyz)//3,1))
        r = run(core,xyz,cur,stamp)
        assert np.array_equal(r['probabilities'],cur)
        assert r['status']['reason'] == reason
        assert r['status']['pairs'] == 0
        if reason == 'point_budget':
            assert core.history is None


def test_ray_depth_gate_rejects_close_foreground_background():
    core = CausalConsistency()
    run(core,XYZ,probs(10,.99),0)
    r = run(core,XYZ + [0.2,0,0],probs(12,.4),500000000)
    assert r['status']['pairs'] == 0  # within Euclidean radius but different ray depth


def test_mutual_matching_does_not_duplicate_one_history_vote():
    core = CausalConsistency()
    run(core,XYZ[:1],probs(10,.99,1),0)
    r = run(core,np.array([[5.,0,0],[5.,.05,0]]),probs(12,.4,2),500000000)
    assert r['status']['pairs'] == 1


def test_pose_revision_reanchors_raw_history_and_missing_anchor_clears_it():
    core = CausalConsistency()
    run(core,XYZ,probs(10,.99),0)
    revised = POSE.copy()
    revised[0,3] = 10
    core.revise_pose([0],[revised])
    r = run(core,XYZ,probs(12,.4),500000000,revised)
    assert r['status']['pairs'] == 3
    core.revise_pose([600000000],[POSE])
    assert core.history is None


def test_caller_mutation_does_not_rewrite_evidence_or_pose():
    core = CausalConsistency()
    p, xyz, pose = probs(10,.99), XYZ.copy(), POSE.copy()
    run(core,xyz,p,0,pose)
    p[:]=0; xyz[:]=50; pose[0,3]=50
    assert core.history['probabilities'][0,10] == np.float32(.99)
    assert np.array_equal(core.history['xyz'],XYZ)
    assert np.array_equal(core.history['pose'],POSE)


def test_reordered_frames_and_invalid_inputs_do_not_corrupt_history():
    core = CausalConsistency()
    run(core,XYZ,probs(10,.99),10)
    for stamp, p, pose in ((10,probs(12,.4),POSE), (11,np.ones((3,16)),POSE),
                           (11,probs(12,.4),np.zeros((4,4)))):
        with pytest.raises(ValueError):
            run(core,XYZ,p,stamp,pose)
        assert core.last_stamp_ns == 10
    core.reset()
    assert run(core,XYZ,probs(10,.99),0)['status']['reason'] == 'no_history'


def test_confident_wrong_history_is_not_mistaken_for_a_correctness_guarantee():
    # Explicit limitation: valid geometry + confident wrong context can hurt a
    # weak but correct current prediction. Real validation must count this harm.
    core = CausalConsistency()
    run(core,XYZ,probs(10,.99),0)
    p = probs(12,.3)
    p[:,10]=.29
    p[:,[i for i in range(16) if i not in (10,12)]]=.41/14
    assert run(core,XYZ,p,500000000)['status']['changed'] == 3


def test_early_veto_equals_full_query_reference_including_duplicate_candidates():
    rng=np.random.default_rng(42)
    xyz=rng.uniform([3,-2,-1],[6,2,1],(300,3))
    xyz[100:120]=xyz[:20]
    current=xyz+rng.normal(0,.025,xyz.shape)
    hp=rng.dirichlet(np.ones(16)*.04,len(xyz)).astype(np.float32)
    cp=rng.dirichlet(np.ones(16)*.1,len(xyz)).astype(np.float32)
    core=CausalConsistency()
    run(core,xyz,hp,0)
    optimized=run(core,current,cp,500000000)['probabilities']
    ci,hi=associate(current,xyz,POSE,POSE,core.cfg,deduplicate=False)
    reference=cp.copy()
    reference[ci],_=fuse(cp[ci],hp[hi],.5,core.cfg)
    assert np.array_equal(optimized,reference)


def test_vetoed_closer_point_remains_in_reciprocal_geometry():
    core=CausalConsistency()
    run(core,XYZ[:1],probs(10,.99,1),0)
    cur=np.array([[5.,0,0],[5.,.05,0]])
    cp=probs(12,.4,2); cp[0]=probs(12,.99,1)[0]
    r=run(core,cur,cp,500000000)
    assert r['status']['candidates']==1
    assert r['status']['pairs']==0  # reverse NN is the vetoed foreground point
    assert np.array_equal(r['probabilities'],cp)


def test_exact_duplicate_queries_preserve_original_indices_and_ties():
    rng=np.random.default_rng(3)
    current=np.r_[np.tile([[.02,0,0]],(1000,1)),rng.uniform(2,10,(100,3))]
    past=current.copy(); past[:1000,0]=0.
    cfg=ConsistencyConfig()
    fast=associate(current,past,POSE,POSE,cfg)
    slow=associate(current,past,POSE,POSE,cfg,deduplicate=False)
    assert all(np.array_equal(a,b) for a,b in zip(fast,slow))
