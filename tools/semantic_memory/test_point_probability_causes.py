import unittest

import numpy as np

from .point_probability_causes import diagnose, online_signals, histogram_update
from .point_probability_memory import mixture, classify


class ProbabilityCauseTests(unittest.TestCase):
    def test_wrong_evidence_survives_correct_correspondence_and_gt_never_relabels_it(self):
        p = np.zeros((3,16),np.float32)
        p[:,0] = [.6,.6,.4]
        p[:,1] = 1-p[:,0]
        old = np.zeros((3,16),np.float32)
        old[:,0] = [.01,.01,.99]
        old[:,1] = 1-old[:,0]
        indices = np.repeat(np.arange(3)[:,None],4,axis=1)
        valid = np.ones_like(indices,bool)
        history = classify(mixture(p,old,indices,valid),p.argmax(1))
        old_copy = old.copy()
        pred, counts = diagnose(p,old,np.array([0,0,0]),np.array([1,2,3]),
            np.array([0,1,0]),np.array([1,2,99]),indices,valid,history)
        # Row 0 is the correct physical object but has wrong semantic evidence.
        # Row 1 borrows the wrong class and is fixed by rejecting that neighbor.
        # Row 2's useful same-class evidence belongs to a different instance.
        np.testing.assert_array_equal(history,[1,1,0])
        np.testing.assert_array_equal(pred['gt_same_class'],[1,0,0])
        np.testing.assert_array_equal(pred['gt_same_object'],[1,0,1])
        np.testing.assert_array_equal(counts['harms_all_neighbors_same_gt_class'],[1,0,0])
        np.testing.assert_array_equal(counts['harms_all_neighbors_same_gt_object'],[1,0,0])
        np.testing.assert_array_equal(counts['gt_same_class_harms_removed'],[0,1,0])
        np.testing.assert_array_equal(counts['gt_same_object_fixes_lost'],[0,0,1])
        np.testing.assert_array_equal(old,old_copy)

    def test_no_valid_neighbors_returns_current_and_unknown_ids_cannot_match(self):
        p = np.eye(16,dtype=np.float32)[[0,10]]
        old = np.eye(16,dtype=np.float32)[[1,11]]
        indices = np.repeat(np.arange(2)[:,None],4,axis=1)
        valid = np.ones_like(indices,bool)
        history = classify(mixture(p,old,indices,valid),p.argmax(1))
        pred,_ = diagnose(p,old,np.array([0,10]),np.array([0,0]),np.array([0,10]),
            np.array([0,0]),indices,valid,history)
        np.testing.assert_array_equal(pred['gt_same_object'],[0,10])
        pred,cat = diagnose(p,old,np.array([0,10]),np.array([1,0]),np.array([0,10]),
            np.array([1,0]),indices,np.zeros_like(valid),p.argmax(1))
        for v in pred.values():
            np.testing.assert_array_equal(v,p.argmax(1))
        self.assertFalse(any(x.any() for x in cat.values()))

    def test_online_signals_exclude_missing_neighbors_and_histogram_conserves_counts(self):
        p = np.eye(16,dtype=np.float32)[[0]]
        old = np.eye(16,dtype=np.float32)[[1,2]]
        indices = np.array([[0,1,1,1]])
        valid = np.array([[1,0,0,0]],bool)
        signals,label = online_signals(p,old,np.array([[0.,0,0]]),np.array([[.1,0,0],[10.,0,0]]),indices,valid)
        self.assertAlmostEqual(signals['mean_distance_m'][0],.1)
        self.assertEqual(signals['history_confidence'][0],1)
        np.testing.assert_array_equal(label,[1])
        storage = {}
        bins = {k:[-1e-6,.5,1.000001] for k in signals}
        histogram_update(storage,signals,np.array([True]),bins)
        for val in storage.values():
            self.assertEqual(val['count'],1)
            self.assertEqual(sum(val['histogram']),1)
        with self.assertRaises(ValueError):
            histogram_update({},dict(x=np.array([1.1])),np.array([True]),dict(x=[0,1]))


if __name__ == '__main__':
    unittest.main()
