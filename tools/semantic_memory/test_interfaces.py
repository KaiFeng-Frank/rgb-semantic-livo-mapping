"""Regression checks for transport, provenance, and ROS geometry contracts."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from ptv3_client import PTv3Client
from tools.semantic_memory.ros_node import read_xyz_intensity, transform_from_odometry


class TransportTests(unittest.TestCase):
    def test_hung_worker_is_killed_at_read_deadline(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, 'rb', buffering=0) as stream:
            client = PTv3Client.__new__(PTv3Client)
            client.p = SimpleNamespace(stdout=stream, kill=Mock())
            client._read_timeout_s = .02
            start = time.monotonic()
            try:
                with self.assertRaises(TimeoutError):
                    client._readn(1)
                client.p.kill.assert_called_once()
                self.assertLess(time.monotonic() - start, 1.)
            finally:
                os.close(write_fd)

    def test_sync_cannot_consume_another_frames_async_result(self):
        client = PTv3Client.__new__(PTv3Client)
        client.pending = [(0, 0, 8)]
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            client.segment(np.zeros((8, 4), np.float32))

    def test_shared_memory_is_unlinked_even_with_exported_views(self):
        client = PTv3Client.__new__(PTv3Client)
        shared = Mock()
        shared.close.side_effect = BufferError('outstanding view')
        client.shm = shared
        client._unlink_shm()
        shared.unlink.assert_called_once()
        self.assertIsNone(client.shm)


class GeometryTests(unittest.TestCase):
    def test_transform_direction_and_invalid_quaternion(self):
        orientation = SimpleNamespace(x=0., y=0., z=np.sqrt(.5), w=np.sqrt(.5))
        message = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=1., y=2., z=3.), orientation=orientation)))
        t = transform_from_odometry(message)
        np.testing.assert_allclose(t @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-6)
        orientation.w = 4.
        with self.assertRaises(ValueError):
            transform_from_odometry(message)

    @unittest.skipUnless(importlib.util.find_spec('sensor_msgs_py'), 'Run with sourced ROS system Python')
    def test_padded_rows_endianness_and_no_input_mutation(self):
        from sensor_msgs.msg import PointCloud2, PointField
        names, offsets = ['x', 'y', 'z', 'intensity'], [0, 4, 8, 16]
        for big in (False, True):
            dtype = np.dtype(dict(names=names, formats=[('>' if big else '<')+'f4']*4,
                                  offsets=offsets, itemsize=24))
            buffer = bytearray(2 * 64)
            rows = np.ndarray((2, 2), dtype=dtype, buffer=buffer, strides=(64, 24))
            expected = np.arange(16).reshape(4, 4).astype(np.float32)
            for i, name in enumerate(names):
                rows[name] = expected[:, i].reshape(2, 2)
            msg = PointCloud2(height=2, width=2, point_step=24, row_step=64,
                fields=[PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                        for n, o in zip(names, offsets)], data=bytes(buffer), is_bigendian=big)
            before = bytes(msg.data)
            np.testing.assert_array_equal(read_xyz_intensity(msg), expected)
            self.assertEqual(bytes(msg.data), before)
            msg.row_step = 16
            with self.assertRaises(ValueError):
                read_xyz_intensity(msg)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch runtime required')
class ExportTests(unittest.TestCase):
    def test_deployment_rejects_untrained_or_incompatible_execution(self):
        import torch
        from ptv3_worker import Segmenter
        model = Segmenter.__new__(Segmenter)
        model.half, model.shuffle, model.torch = True, False, torch
        model.fast_voxel = model.gpu_voxel = model.fast_hilbert = False
        config = dict(backbone=dict(patch_size=1024), num_classes=16)
        model.cfg = SimpleNamespace(model=SimpleNamespace(to_dict=lambda: config))
        before = model.memory_model_sha256()
        config['backbone']['patch_size'] = 512
        self.assertNotEqual(before, model.memory_model_sha256())
        with tempfile.TemporaryDirectory() as temporary:
            from semantic_memory import MemoryConfig
            from dataclasses import asdict
            path = Path(temporary)/'synthetic.pt'
            for trained, deployable in ((False, False), (True, False), (1, True)):
                torch.save(dict(schema=1, config=asdict(MemoryConfig()), trained=trained,
                                provenance=dict(scope='real_training', deployable=deployable)), path)
                with self.assertRaisesRegex(ValueError, 'real-training'):
                    model.enable_memory(path)
        model.shuffle = True
        with self.assertRaisesRegex(ValueError, 'shuffle=False'):
            model.enable_memory(shadow=True)

    def test_export_preserves_synthetic_flag_and_consumable_metadata(self):
        import torch
        from tools.semantic_memory.cache import export
        from tools.semantic_memory.train import load_manifest, load_frame
        class Extractor:
            num_classes, grid_size, intensity_scale = 16, .05, .2
            model = SimpleNamespace(seg_head=SimpleNamespace(in_features=64))
            def __init__(self, **kwargs):
                pass
            def memory_model_sha256(self):
                return 'c' * 64
            def segment(self, points, **kwargs):
                n = len(points)
                return dict(xyz=torch.from_numpy(points[:, :3].copy()), features=torch.ones(n, 64),
                            logits=torch.zeros(n, 16), inverse=torch.arange(n))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'weights').write_bytes(b'fixture, not a trained backbone')
            np.array([[1, 2, 3, .5], [2, 3, 4, .1]], np.float32).tofile(root/'scan.bin')
            np.save(root/'labels.npy', np.array([2, 3], np.int64))
            spec = dict(schema=1, geometry='deskewed_sensor', label_space='native16',
                        split='train', synthetic=True, sequences=[dict(id='00', frames=[
                            dict(frame=0, stamp=0., path='scan.bin', labels='labels.npy',
                                 transform=np.eye(4).tolist())])])
            source = root/'source.json'
            source.write_text(json.dumps(spec))
            with patch('ptv3_worker.Segmenter', Extractor), patch('ptv3_loader_verified.CKPT_PATH', str(root/'weights')):
                export(source, root/'cache')
                manifest = load_manifest(root/'cache/manifest.json')
                self.assertTrue(manifest.synthetic)
                raw = load_frame(manifest, manifest.frames[0])
                self.assertEqual(raw['label'].tolist(), [2, 3])
                spec['synthetic'] = 'true'
                source.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError, 'boolean'):
                    export(source, root/'invalid')
                self.assertFalse((root/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
