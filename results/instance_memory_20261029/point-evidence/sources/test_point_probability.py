import unittest
import numpy as np
from .point_probability_memory import RawProbabilityMemory,mixture,classify


class ProbabilityTests(unittest.TestCase):
    def test_per_frame_mass_is_invariant_to_duplicate_raw_probabilities(self):
        p=np.array([[.7,.3]+[0.]*14],np.float32);h=np.array([[.1,.9]+[0.]*14],np.float32)
        j=np.zeros((1,4),int)
        single=mixture(p,h,j,np.array([[1,0,0,0]],bool));four=mixture(p,h,j,np.ones((1,4),bool))
        np.testing.assert_allclose(single,four);np.testing.assert_allclose(single[0,:2],[.4,.6])
        np.testing.assert_array_equal(mixture(p,h,j,np.zeros((1,4),bool)),p)
        tied=np.array([[.5,.5]+[0.]*14],np.float32)
        np.testing.assert_array_equal(classify(tied,np.array([1])),1)

    def test_memory_keeps_raw_distribution_after_output_changed(self):
        m=RawProbabilityMemory();xyz=np.zeros((4,3));old=np.zeros((4,16),np.float32);old[:,1]=1
        current=old*.8;current[:,1]=.4;current[:,2]=.6
        m.step(xyz,old,0,'a');out,_=m.step(xyz,current,500000000,'a')
        np.testing.assert_array_equal(out['previous_raw_probability'],1)
        np.testing.assert_array_equal(m.previous['p'],current)
        np.testing.assert_array_equal(m.step(xyz,current,1500000000,'a')[0]['previous_raw_probability'],2)
        with self.assertRaises(ValueError):m.step(xyz,current,1500000000,'a')
        np.testing.assert_array_equal(m.step(xyz,current,0,'b')[0]['previous_raw_probability'],2)


if __name__=='__main__':unittest.main()
