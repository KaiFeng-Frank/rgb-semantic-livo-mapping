import unittest
import numpy as np
from frontend.online_lio_validation import pose_delivery_latency


class DeliveryTiming(unittest.TestCase):
    def setUp(self):
        self.trajectory=np.array([[1.05,1.,2.,3.,0,0,0,1],[1.15,4.,5.,6.,0,0,0,1]])
        self.odom=[dict(received=10.2,xyz=[1.,2.,3.]),dict(received=10.31,xyz=[4.,5.,6.])]
        self.raw=[(1,10.),(2,10.1)]
        self.ends=np.array([1_100_000_000,1_200_000_000])

    def test_sensor_time_uses_first_complete_containing_scan(self):
        np.testing.assert_allclose(pose_delivery_latency(self.trajectory,self.odom,self.raw,self.ends),[200,210])

    def test_do_not_pair_dropped_or_reordered_odometry(self):
        self.assertIsNone(pose_delivery_latency(self.trajectory,self.odom[:1],self.raw,self.ends))
        self.assertIsNone(pose_delivery_latency(self.trajectory,self.odom[::-1],self.raw,self.ends))

    def test_out_of_coverage_pose_is_not_a_latency_measurement(self):
        self.trajectory[-1,0]=1.3
        self.assertIsNone(pose_delivery_latency(self.trajectory,self.odom,self.raw,self.ends))


if __name__=='__main__':unittest.main()
