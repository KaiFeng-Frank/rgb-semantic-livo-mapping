import unittest
from frontend.input_bridge import DeadlineBuffer, add_fields


class Timing(unittest.TestCase):
    def test_fixed_delay_control(self):
        b=DeadlineBuffer(80.,fixed_delay=True)
        b.push(1,'scan',0.)
        b.complete(1,'labels',.01)
        self.assertEqual(b.drain(.02),[])
        self.assertEqual(b.drain(.08)[0].result,'labels')

    def test_arrival_order_and_once_only(self):
        b = DeadlineBuffer(80.)
        b.push(1, 'first', 0.)
        b.push(2, 'second', .01)
        b.complete(2, 'label2', .02)
        self.assertEqual(b.drain(.02), [])
        b.complete(1, 'label1', .03)
        self.assertEqual([p.key for p in b.drain(.03)], [1, 2])
        self.assertEqual(b.drain(.2), [])
        self.assertFalse(b.complete(1, 'late', .04))

    def test_late_inference_does_not_refuse_geometry(self):
        b = DeadlineBuffer(50.)
        b.push(1, 'scan', 1.)
        self.assertFalse(b.complete(1, 'labels', 1.051))
        p, = b.drain(1.051)
        self.assertIsNone(p.result)
        self.assertEqual(p.payload, 'scan')

    def test_bounded_backpressure_and_duplicate(self):
        b = DeadlineBuffer(100., capacity=1)
        b.push(1, 'scan', 0.)
        with self.assertRaises(BufferError): b.push(2, 'next', .01)
        self.assertEqual(len(b.drain(.01, force_oldest=True)), 1)
        with self.assertRaises(ValueError): b.push(1, 'duplicate', .02)
        b.push(2, 'next', .02)

    def test_completion_timestamp_not_timer_timestamp(self):
        b = DeadlineBuffer(50.)
        b.push(1, None, 0.)
        self.assertTrue(b.complete(1, 'labels', .049))
        self.assertEqual(b.drain(.09)[0].result, 'labels')


class ROSFields(unittest.TestCase):
    def test_preserves_raw_bytes_and_organized_padding(self):
        try:
            from sensor_msgs.msg import PointCloud2, PointField
        except ImportError:
            self.skipTest('source ROS 2 setup to test PointCloud2 transport')
        import numpy as np
        msg = PointCloud2(height=2, width=2, point_step=16, row_step=36,
                         fields=[PointField(name='x', offset=0, datatype=7, count=1)],
                         data=bytes(range(72)))
        msg.header.frame_id='lidar'
        msg.header.stamp.sec=42
        out = add_fields(msg, [10,12,6,999], [.9,.8,.7,float('nan')])
        a=np.frombuffer(out.data,dtype=np.uint8).reshape(4,24)
        original=np.ndarray((2,2,16),buffer=msg.data,dtype=np.uint8,strides=(36,16,1)).reshape(4,16)
        np.testing.assert_array_equal(a[:,:16],original)
        self.assertEqual(out.header,msg.header)
        np.testing.assert_array_equal(a[:,16:18].copy().view('<u2').ravel(),[10,12,6,65535])
        self.assertEqual(out.point_step,24)


if __name__=='__main__': unittest.main()
