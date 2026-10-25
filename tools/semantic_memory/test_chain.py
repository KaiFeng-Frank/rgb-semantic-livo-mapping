"""Evidence provenance, causal diagnostics and recognition-coverage checks."""
import copy
import unittest

import numpy as np
import torch

from tools.semantic_memory.chain_trace import TracedEvidence, FusionCapture, exact_source_indices, voxel_instances, candidate_tags
from tools.semantic_memory.chain_metrics import aggregate
from tools.semantic_memory.chain_audit import supply_positive_raw
from semantic_memory import MemoryConfig
from semantic_repair import CurrentAnchoredRepair, RepairConfig


class ChainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.cfg = MemoryConfig(feature_dim=4,classes=3,cell_size=1.5,radius=1.5,
            neighbors=4,records_per_cell=4,min_cosine=.7,max_age_s=.75,capacity=16)
        self.repair = RepairConfig(moving_classes=(1,),max_queries=16)

    def raw(self,frame):
        return dict(xyz_sensor=torch.tensor([[5.+frame,0,0],[5.1+frame,0,0],[8.,1.,0.]]),
            features=torch.tensor([[1.,.3,0.,.1],[1.,.2,.1,0.],[0.,0.,1.,1.]]),
            base_logits=torch.tensor([[0.,.6,0.],[0.,.4,0.],[.5,0.,0.]]),
            anchor_logits=torch.tensor([[.3,.5,0.],[.3,.4,0.],[.5,.1,0.]]),
            stamp=frame*.5,frame=frame,transform=torch.eye(4))

    def test_instrumentation_preserves_prediction_and_tags_selected_raw_rows(self):
        model = CurrentAnchoredRepair(self.cfg,self.repair).eval()
        with torch.no_grad():
            model.fusion.net[-1].weight.normal_(0,.2)
        other = copy.deepcopy(model)
        model.memory = TracedEvidence(self.cfg,'cpu',self.repair.moving_classes)
        capture = FusionCapture(model.fusion)
        for frame in range(3):
            raw = self.raw(frame)
            actual,expected = model(**raw),other(**raw)
            self.assertTrue(torch.equal(actual['logits'],expected['logits']))
            self.assertTrue(torch.equal(model.memory.xyz,raw['xyz_sensor'][model.memory.raw_index]))
            self.assertTrue(torch.equal(model.memory.features,raw['features'][model.memory.raw_index].half()))
            if frame:
                self.assertTrue((model.memory.trace['record_stamp']<raw['stamp']).all())
                self.assertEqual(capture.value[2]['valid'].shape,(3,4))
        capture.close()

    def test_source_provenance_rejects_ambiguity(self):
        points=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        self.assertEqual(exact_source_indices(points,points[[1,0]]).tolist(),[1,0])
        with self.assertRaises(ValueError):
            exact_source_indices(points[[0,0]],points[[0]])

    def test_voxel_identity_excludes_mixed_objects_and_unassigned_points(self):
        arrays=dict(xyz=np.zeros((4,3)),inverse=np.array([0,0,1,1,2,2,3]),
            point_label=np.array([3,3,3,3,3,3,-1]),group_owner=np.array([0,0,0,1,0,-1,-1]))
        groups=[dict(instance='a'),dict(instance='b')]
        result=voxel_instances(arrays,groups,{'a':5,'b':6})
        self.assertEqual(result.tolist(),[5,-1,-1,-1])

    def test_diagnostic_supply_keeps_radius_cosine_and_original_candidates_unchanged(self):
        empty=dict(features=torch.zeros(2,4,4),probabilities=torch.zeros(2,4,3),
            delta=torch.zeros(2,4,3),age=torch.zeros(2,4),valid=torch.zeros(2,4,dtype=torch.bool))
        previous=dict(instances=torch.tensor([5,6]),prediction=torch.tensor([1,1]),labels=torch.tensor([1,1]),
            world=torch.tensor([[0.,0,0],[.1,0,0]]),features=torch.tensor([[1.,0,0,0],[1.,0,0,0]]),
            probabilities=torch.tensor([[0.,1.,0.],[0.,1.,0.]]),stamp=0.,frame=0)
        trace=dict(query_xyz=torch.tensor([[.1,0,0],[2.,0,0]]),query_radius=torch.ones(2),
            query_features=torch.tensor([[1.,0,0,0],[1.,0,0,0]]),stamp=.5)
        result=supply_positive_raw(empty,trace,previous,torch.tensor([5,5]),{5},self.cfg)
        self.assertTrue(result['valid'][0,0])
        self.assertFalse(result['valid'][1].any())
        self.assertTrue(torch.allclose(result['delta'][0,0],torch.tensor([-.1,0,0])))
        self.assertFalse(empty['valid'].any())
        trace['query_features'][0]=torch.tensor([0.,1.,0.,0.])
        self.assertFalse(supply_positive_raw(empty,trace,previous,torch.tensor([5,5]),{5},self.cfg)['valid'].any())
        trace['stamp']=0.
        with self.assertRaises(ValueError):
            supply_positive_raw(empty,trace,previous,torch.tensor([5,5]),{5},self.cfg)

    def test_any_hit_is_not_enough_to_recognize_an_object(self):
        r=dict(scene='s',label=6,previous_raw_correct_fraction=1.,
            correct_fraction=dict(anchor=0.,no_history=0.,temporal=0.,clean_existing_gt=0.,supply_raw_gt=1.),
            raw_good_voxels=5,stored_good_voxels=2,readable_good_voxels=2,
            points_needed=3,confidence_error_points=5,queried_error_points=4,bounded_reachable_error_points=4,
            local_good_error_points=2,shortlist_good_error_points=1,retrieved_good_error_points=1,
            supplied_good_error_points=4)
        result=aggregate([r])
        self.assertEqual(result['funnel']['retrieved'],1)
        self.assertEqual(result['funnel']['repaired'],0)
        self.assertEqual(result['totals']['retrieved_good_error_points_enough_for_recognition'],0)
        self.assertEqual(result['totals']['supplied_good_error_points_enough_for_recognition'],1)
        self.assertEqual(result['first_loss'],{'repaired':1})


if __name__=='__main__':
    unittest.main()
