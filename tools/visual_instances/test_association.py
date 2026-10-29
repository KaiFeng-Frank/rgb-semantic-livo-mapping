"""Failure cases for association, transport, and identity lifetime."""
import json
import unittest

import numpy as np

from .geometry import Camera, FrameCache, InstanceFrame, associate, mask_frame, camera_at_scan


class AssociationTests(unittest.TestCase):
    def frame(self, stamp=100, session='a'*32):
        index = np.zeros((40, 40), np.uint16)
        index[5:20, 5:20] = 1
        index[5:20, 22:35] = 2
        return InstanceFrame(stamp, session, index, np.array([12, 17]), np.array([6, 6]), np.array([.9, .8]))

    def camera(self):
        return Camera(40, 40, np.diag([10., 10., 1.]), np.zeros(4), np.eye(4))

    def test_two_adjacent_people_and_occluding_surface(self):
        xyz = np.array([[5,5,5], [5.05,5,5], [5,5.05,5],
                        [12.5,5,5], [12.55,5,5], [12.5,5.05,5],
                        [10,10,10], [5,5,-5], [100,100,5], [0,0,5]])
        out = associate(xyz, self.camera(), self.frame())
        self.assertEqual(out['ids'].tolist(), [12,12,12,17,17,17,0,0,0,0])
        self.assertEqual(out['info']['occluded'], 1)
        self.assertEqual(out['info']['instances'], 2)

    def test_sparse_object_abstains(self):
        out = associate(np.array([[5,5,5]]), self.camera(), self.frame())
        self.assertEqual(out['ids'].tolist(), [0])

    def test_calibration_uses_camera_not_lidar_depth(self):
        cam = self.camera()
        cam.T_camera_lidar = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1.]])
        out = associate(np.array([[5,5,5], [5,5.05,5], [5,5,5.05]]), cam, self.frame())
        self.assertEqual(out['ids'].tolist(), [12,12,12])

    def test_roundtrip_and_corrupt_packet(self):
        f = self.frame()
        g = InstanceFrame.decode(f.encode())
        np.testing.assert_array_equal(g.index, f.index)
        np.testing.assert_array_equal(g.ids, f.ids)
        d = json.loads(f.encode()); d['shape'] = [40000, 40000]
        with self.assertRaises(ValueError): InstanceFrame.decode(json.dumps(d))
        d = json.loads(f.encode()); d['ids'] = [12.5,17]
        with self.assertRaises(ValueError): InstanceFrame.decode(json.dumps(d))
        d = json.loads(f.encode()); d['ids'] = [12,12]
        with self.assertRaises(ValueError): InstanceFrame.decode(json.dumps(d))
        d = json.loads(f.encode()); d['shape'] = [20,20]
        with self.assertRaises(ValueError): InstanceFrame.decode(json.dumps(d))

    def test_time_gate_never_uses_future_and_never_waits(self):
        c = FrameCache(max_dt_ms=20)
        c.add(self.frame(100_000_000))
        self.assertIsNone(c.get(99_000_000))
        self.assertIsNotNone(c.get(110_000_000))
        self.assertIsNone(c.get(121_000_000))

    def test_frozen_features_survive_wire_without_changing_association(self):
        f=self.frame();f.visual_features=np.arange(128,dtype=np.float32).reshape(2,64)
        f.visual_feature_valid=np.array([True,False])
        g=InstanceFrame.decode(f.encode())
        np.testing.assert_array_equal(g.visual_features,f.visual_features)
        np.testing.assert_array_equal(g.visual_feature_valid,f.visual_feature_valid)
        xyz=np.array([[5,5,5],[5.05,5,5],[5,5.05,5]])
        out=associate(xyz,self.camera(),g)
        np.testing.assert_array_equal(out['ids'],associate(xyz,self.camera(),self.frame())['ids'])
        np.testing.assert_array_equal(out['visual_features'][12][0],f.visual_features[0])
        d=json.loads(f.encode());d['visual_features'][0][0]=float('nan')
        with self.assertRaises(ValueError):InstanceFrame.decode(json.dumps(d))

    def test_restart_does_not_reuse_old_identity(self):
        c = FrameCache()
        c.add(self.frame())
        c.add(self.frame(200, 'b'*32))
        self.assertEqual(len(c.frames), 1)
        with self.assertRaises(ValueError): c.add(self.frame(300))
        with self.assertRaises(ValueError): c.add(self.frame(200, 'b'*32))

    def test_invalid_mask_and_camera(self):
        with self.assertRaises(ValueError):
            Camera(40,40,np.diag([-10.,10.,1.]),np.zeros(4),np.eye(4))
        f = self.frame(); f.index = f.index[:20]
        with self.assertRaises(ValueError): associate(np.array([[5,5,5]]), self.camera(), f)

    def test_overlaps_with_rejected_foreground_and_boundaries_abstain(self):
        masks=np.zeros((2,40,40),bool)
        masks[0,5:25,5:25]=True
        masks[1,15:35,15:35]=True
        f=mask_frame(masks,[1,2],[0,2],[.95,.2],100,'a'*32,(40,40))
        self.assertEqual(f.index[10,10],1)
        self.assertEqual(f.index[5,10],0)
        self.assertEqual(f.index[18,18],0)  # Low-score foreground still vetoes overlap.
        self.assertEqual(f.index[30,30],0)
        self.assertEqual(f.labels.tolist(),[6])

    def test_empty_detection_is_explicit_unknown(self):
        f=mask_frame(np.zeros((0,40,40),bool),[],[],[],100,'a'*32,(40,40))
        g=InstanceFrame.decode(f.encode())
        self.assertEqual(len(g.ids),0)
        self.assertEqual(np.count_nonzero(g.index),0)

    def test_camera_ego_motion_uses_only_received_pose_bracket(self):
        poses=np.stack([np.eye(4),np.eye(4)])
        poses[1,0,3]=2
        graph=dict(times_ns=np.array([10**9,2*10**9]),poses=poses)
        c=camera_at_scan(self.camera(),graph,2*10**9,1_500_000_000,np.eye(4))
        np.testing.assert_allclose(c.T_camera_lidar[:3,3],[1,0,0])
        self.assertIsNone(camera_at_scan(self.camera(),graph,2*10**9,999_000_000,np.eye(4)))
        self.assertIsNone(camera_at_scan(self.camera(),graph,2*10**9,3*10**9,np.eye(4)))


if __name__ == '__main__':
    unittest.main()
