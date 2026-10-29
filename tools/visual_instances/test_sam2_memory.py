import unittest
from types import SimpleNamespace
import torch
from .sam2_memory import SAM2Pair


class SAM2StateTests(unittest.TestCase):
    def test_bound_spatial_memory_retain_pointers_and_owned_storage(self):
        m=SAM2Pair.__new__(SAM2Pair);m.model=SimpleNamespace(num_maskmem=7,max_obj_ptrs_in_encoder=16);m.histories={}
        value=dict(obj_ptr=torch.ones((2,4)),maskmem_features=torch.ones((2,2,3,3)),maskmem_pos_enc=[torch.ones((2,2,3,3))])
        for f in range(30):
            m.frame=f
            for slot,key in enumerate([10,20]):m._retain(key,value,slot,is_new=f==0)
        h=m.histories[10]
        self.assertEqual(list(h['cond_frame_outputs']),[0]);self.assertEqual(len(h['non_cond_frame_outputs']),16)
        self.assertEqual(sum(r['maskmem_features'] is not None for r in h['non_cond_frame_outputs'].values()),6)
        self.assertTrue(all(r['obj_ptr'].shape==(1,4) for r in h['non_cond_frame_outputs'].values()))
        value['obj_ptr'].zero_();self.assertEqual(float(h['cond_frame_outputs'][0]['obj_ptr'].sum()),4.)
        merged=m._batched_history([10,20]);self.assertEqual(merged['cond_frame_outputs'][0]['maskmem_features'].shape[0],2)
        self.assertIsNone(merged['non_cond_frame_outputs'][14]['maskmem_features'])


if __name__=='__main__':unittest.main()
