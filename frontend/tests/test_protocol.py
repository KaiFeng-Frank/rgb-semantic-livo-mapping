import json
from pathlib import Path
import tempfile
import unittest
from frontend.protocol import load, raw_dir, REPO

class Protocol(unittest.TestCase):
    def test_cross_sequence_source_is_frozen(self):
        # The old experiment remains frozen to its own estimator. A revised
        # implementation must not silently reuse it with updated source hashes.
        cfg=json.loads((REPO/'frontend/holdout09.json').read_text())
        self.assertEqual(raw_dir(Path('/inputs'),cfg).name,'2011_09_30_drive_0033_sync')
        self.assertEqual(len(cfg['frozen_sources']),5)
        self.assertEqual(cfg['repetitions'],3)
        with self.assertRaisesRegex(ValueError,'frozen estimator changed'):
            load(REPO/'frontend/holdout09.json')

    def reject(self, mutate):
        cfg=json.loads((REPO/'frontend/holdout09.json').read_text());mutate(cfg)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'plan.json';p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load(p)

    def test_reject_unapproved_large_window(self):
        self.reject(lambda c:c['short_sequence'].update(end_exclusive=1000))

    def test_reject_mismatched_estimator(self):
        self.reject(lambda c:c['frozen_sources'].update({'frontend/include/semantic_frontend.hpp':'0'*64}))

    def test_old_protocol_still_selects_development_sequence(self):
        cfg=load()
        self.assertEqual(raw_dir(Path('/inputs'),cfg).name,'2011_09_30_drive_0027_sync')
        self.assertEqual(cfg['short_sequence']['end_exclusive'],120)

    def test_enhanced_window_is_bounded_and_cannot_be_scored(self):
        cfg=load(REPO/'frontend/enhanced-diagnostic.json')
        self.assertTrue(cfg['diagnostic'])
        self.assertEqual(cfg['short_sequence'],dict(sequence='07',first_frame=730,end_exclusive=826))
        self.assertEqual(cfg['repetitions'],1)

if __name__=='__main__':unittest.main()
