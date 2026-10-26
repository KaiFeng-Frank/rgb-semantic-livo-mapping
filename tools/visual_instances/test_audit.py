"""Distinguish changing a class from changing the physical object behind an ID."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .audit import AssociationAudit
from .geometry import InstanceFrame


class AuditTests(unittest.TestCase):
    def observe(self, audit, i, label, tokens, object_ids):
        frame = InstanceFrame(i, 'a'*32, np.ones((2, 2), np.uint16),
                              np.array([7]), np.array([label]), np.array([.9]))
        row = dict(objects=[dict(instance=t, label=3, translation=[0, 0, 0], size=[1, 1, 1]) for t in tokens])
        audit.step('scene', i, row, dict(image_path='fixture', image_sha256='fixture'), frame,
                   dict(ids=np.full(4, 7)), np.full(4, 3), np.asarray(object_ids), {})

    def test_class_change_and_identity_change_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AssociationAudit(Path(tmp))
            self.observe(audit, 0, 3, ['car-A'], [0]*4)
            self.observe(audit, 1, 9, ['car-A'], [0]*4)
            self.observe(audit, 2, 3, ['car-B'], [0]*4)
            counts = audit.finish()
            self.assertEqual(counts['adjacent_track_pairs'], 2)
            self.assertEqual(counts['wrong_identity_pairs'], 1)
            self.assertEqual(counts['same_identity_visual_label_changes'], 1)
            self.assertEqual(counts['same_identity_visual_correct_to_wrong'], 1)
            self.assertFalse(audit.pairs[1]['same_gt_object'])
            self.assertTrue(audit.pairs[0]['same_gt_object'])

    def test_tied_proxy_does_not_become_strong_identity_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AssociationAudit(Path(tmp))
            self.observe(audit, 0, 3, ['car-A', 'car-B'], [0, 1, 0, 1])
            self.observe(audit, 1, 3, ['car-A', 'car-B'], [1, 1, 1, 1])
            counts = audit.finish()
            self.assertEqual(counts['wrong_identity_pairs'], 1)  # Original proxy retained.
            self.assertEqual(counts['wrong_identity_with_tied_vote'], 1)
            self.assertEqual(counts['wrong_identity_strong_box_support'], 0)
            self.assertEqual(counts['wrong_identity_both_visual_labels_correct'], 1)


if __name__ == '__main__':
    unittest.main()
