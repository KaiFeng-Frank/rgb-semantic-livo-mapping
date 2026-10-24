"""A bag's carried point identity is authoritative, including duplicate xyz."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from kitti_to_ros2bag import PC_DTYPE, INDEXED_PC_DTYPE, make_pointcloud2
from frontend.replay_smoke import raw_point_order


class PointIdentity(unittest.TestCase):
    def fixture(self):
        # Duplicate coordinates deliberately forbid recovering IDs from nearest xyz.
        points=np.array([[3,0,-1,.2],[4,1,-1,.3],[3,0,-1,.2],[5,2,-1,.4]],np.float32)
        order=np.array([2,0,3,1],np.uint32)
        arr=np.zeros(4,dtype=INDEXED_PC_DTYPE)
        for j,k in enumerate(('x','y','z','intensity')):arr[k]=points[order,j]
        arr['time']=[0,10,20,30];arr['ring']=[2,0,3,1];arr['raw_point_index']=order
        return points,order,arr

    def test_existing_sensor_bytes_are_unchanged(self):
        points,order,arr=self.fixture()
        old=np.empty(4,dtype=PC_DTYPE)
        for name in old.dtype.names:old[name]=arr[name]
        before,after=make_pointcloud2(42,old,'lidar'),make_pointcloud2(42,arr,'lidar')
        np.testing.assert_array_equal(np.frombuffer(after.data,np.uint8).reshape(4,26)[:,:22],
                                      np.frombuffer(before.data,np.uint8).reshape(4,22))
        self.assertEqual(before.header,after.header)
        np.testing.assert_array_equal(raw_point_order(after,points),order)

    def test_bag_roundtrip_preserves_original_ids(self):
        import rosbag2_py
        from rclpy.serialization import serialize_message,deserialize_message
        from sensor_msgs.msg import PointCloud2
        points,order,arr=self.fixture()
        with tempfile.TemporaryDirectory(prefix='semantic-identity-') as temp:
            storage=rosbag2_py.StorageOptions(uri=temp+'/bag',storage_id='mcap')
            conversion=rosbag2_py.ConverterOptions('','')
            writer=rosbag2_py.SequentialWriter();writer.open(storage,conversion)
            writer.create_topic(rosbag2_py.TopicMetadata(id=0,name='/points',type='sensor_msgs/msg/PointCloud2',serialization_format='cdr'))
            writer.write('/points',serialize_message(make_pointcloud2(42,arr,'lidar')),42)
            del writer
            reader=rosbag2_py.SequentialReader();reader.open(storage,conversion)
            _,data,_=reader.read_next();msg=deserialize_message(data,PointCloud2)
            np.testing.assert_array_equal(raw_point_order(msg,points),order)

    def test_duplicate_or_mismatched_ids_are_rejected(self):
        points,order,arr=self.fixture()
        arr['raw_point_index'][0]=arr['raw_point_index'][1]
        with self.assertRaisesRegex(ValueError,'permutation'):
            raw_point_order(make_pointcloud2(42,arr,'lidar'),points)
        arr['raw_point_index']=order[::-1]
        with self.assertRaisesRegex(ValueError,'xyz/intensity'):
            raw_point_order(make_pointcloud2(42,arr,'lidar'),points)


if __name__=='__main__':unittest.main()
