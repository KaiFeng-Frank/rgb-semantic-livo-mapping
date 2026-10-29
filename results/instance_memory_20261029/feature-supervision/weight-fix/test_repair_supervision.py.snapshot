import unittest
import numpy as np
from .repair_supervision import present_class_weights


class ClassWeightTests(unittest.TestCase):
    def test_unobserved_categories_do_not_change_observed_weights(self):
        counts=np.array([10000,100])
        base=present_class_weights(counts)
        expanded=present_class_weights(np.r_[counts,np.zeros(14)])
        np.testing.assert_array_equal(base,expanded[:2])
        np.testing.assert_array_equal(expanded[2:],0)
        self.assertGreater(base[1]/base[0],7)

    def test_all_present_preserves_existing_formula_and_sample_scaling(self):
        counts=np.array([10000,100,20])
        inverse=np.sqrt(counts.sum()/counts)
        np.testing.assert_array_equal(present_class_weights(counts),np.clip(inverse/inverse.mean(),.25,4))
        np.testing.assert_allclose(present_class_weights(counts),present_class_weights(counts*10),rtol=0,atol=1e-15)

    def test_invalid_or_unobserved_training_pool_is_rejected(self):
        for counts in ([],[0,0],[-1,10],[float('nan'),1],[float('inf'),1],[[1,2]]):
            with self.assertRaises(ValueError):present_class_weights(counts)


if __name__=='__main__':unittest.main()
