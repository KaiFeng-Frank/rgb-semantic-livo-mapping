"""Independent official-evaluator parity and submission integrity tests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .evaluate import Confusion, evaluate
from .schema import VENDOR, RAW_IDS, FRAME_COUNTS, bridge, learning_to_raw, raw_to_learning, taxonomy_audit, digest, save, verify_vendor
from .submission import build_submission, TEST_SEQUENCES


class Contracts(unittest.TestCase):
    def test_matches_pinned_official_evaluator(self):
        verify_vendor()
        spec = importlib.util.spec_from_file_location('official_iou', VENDOR / 'np_ioueval.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with contextlib.redirect_stdout(io.StringIO()):
            official = mod.iouEval(20, [0])
        rng = np.random.default_rng(20261024)
        ours = Confusion()
        for n in [1, 2000, 12345]:
            pred = rng.integers(0, 20, n)
            gt = rng.integers(0, 20, n)
            ours.add(pred, gt)
            official.addBatch(pred, gt)
        np.testing.assert_array_equal(ours.matrix, official.conf_matrix)
        mean, classes = official.getIoU()
        self.assertAlmostEqual(ours.report()['miou19_percent'], mean * 100, places=12)
        np.testing.assert_allclose(list(ours.report()['class_iou_percent'].values()), classes[1:] * 100)

    def test_abstention_is_false_negative_and_all_19_classes_count(self):
        c = Confusion()
        c.add(np.array([1, 0, 19]), np.array([1, 1, 0]))
        r = c.report()
        self.assertEqual(r['evaluated_points'], 2)
        self.assertEqual(r['abstentions_on_evaluated_points'], 1)
        self.assertAlmostEqual(r['class_iou_percent']['car'], 50)
        self.assertAlmostEqual(r['miou19_percent'], 50 / 19)

    def test_moving_gt_instance_bits_and_unknown_ids(self):
        raw = np.array([(7 << 16) | 252, 253, 254, 255, 258, 259], dtype=np.uint32)
        np.testing.assert_array_equal(raw_to_learning(raw), [1, 7, 6, 8, 4, 5])
        with self.assertRaises(ValueError):
            raw_to_learning(raw, prediction=True)
        with self.assertRaises(ValueError):
            raw_to_learning(np.array([60000], dtype=np.uint32))
        np.testing.assert_array_equal(raw_to_learning(learning_to_raw(np.arange(20)), prediction=True), np.arange(20))

    def test_taxonomy_does_not_claim_native_19(self):
        a = taxonomy_audit()
        self.assertFalse(a['native_19_class_model'])
        self.assertEqual(a['reachable_classes'], 13)
        self.assertEqual(len(a['unrepresentable_classes']), 6)
        self.assertEqual(bridge(np.array([7])).item(), 0)
        with self.assertRaises(ValueError):
            bridge(np.array([16]))

    def test_validation_rejects_incomplete_and_test_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                evaluate(Path(tmp), Path(tmp), '08')
            with self.assertRaises(ValueError):
                evaluate(Path(tmp), Path(tmp), '11')

    def fixture(self, root):
        dataset, pred = root / 'data', root / 'pred'
        records = []
        for seq in TEST_SEQUENCES:
            scan = dataset / 'sequences' / seq / 'velodyne' / '000000.bin'
            scan.parent.mkdir(parents=True)
            np.zeros((19, 4), dtype='<f4').tofile(scan)
            rel = f'sequences/{seq}/predictions/000000.label'
            p = pred / rel
            p.parent.mkdir(parents=True)
            RAW_IDS[1:].astype('<u4').tofile(p)
            records.append(dict(path=rel, sha256=digest(p), points=19))
        meta = dict(task='single_scan_semantic_segmentation', num_classes=19,
                    label_space='SemanticKITTI19', uses_future_scans=False,
                    map_fusion=False, frozen_bridge=False, sequences=list(TEST_SEQUENCES),
                    checkpoint_sha256='fixture', training_data_disclosure='synthetic test fixture',
                    predictions=records)
        manifest = root / 'manifest.json'
        save(manifest, meta)
        return dataset, pred, manifest, meta

    def test_submission_passes_official_validator_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(FRAME_COUNTS, {s: 1 for s in TEST_SEQUENCES}):
            root = Path(tmp)
            data, pred, manifest, meta = self.fixture(root)
            result = build_submission(data, pred, manifest, root / 'valid.zip')
            self.assertEqual(result['frames'], 11)
            check = subprocess.run([sys.executable, str(VENDOR / 'validate_submission.py'),
                                    '--task', 'segmentation', str(root / 'valid.zip'), str(data)],
                                   text=True, capture_output=True, check=True)
            self.assertIn('Valid', check.stdout)
            self.assertNotIn('Error', check.stdout)
            p = pred / meta['predictions'][0]['path']
            values = np.fromfile(p, '<u4'); values[0] = 30; values.tofile(p)
            with self.assertRaisesRegex(ValueError, 'changed'):
                build_submission(data, pred, manifest, root / 'changed.zip')

    def test_submission_rejects_bridge_future_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(FRAME_COUNTS, {s: 1 for s in TEST_SEQUENCES}):
            root = Path(tmp)
            data, pred, manifest, meta = self.fixture(root)
            for key in ['frozen_bridge', 'uses_future_scans', 'map_fusion']:
                bad = dict(meta); bad[key] = True; save(manifest, bad)
                with self.assertRaisesRegex(ValueError, 'contract'):
                    build_submission(data, pred, manifest, root / 'bad.zip')
            save(manifest, meta)
            (pred / meta['predictions'][0]['path']).unlink()
            with self.assertRaisesRegex(ValueError, 'frame set'):
                build_submission(data, pred, manifest, root / 'missing.zip')

    def test_truncated_dataset_does_not_make_submission_complete(self):
        self.assertEqual(sum(FRAME_COUNTS[s] for s in TEST_SEQUENCES), 20351)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, pred, manifest, meta = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'scans'):
                build_submission(data, pred, manifest, root / 'truncated.zip')


if __name__ == '__main__':
    unittest.main()
