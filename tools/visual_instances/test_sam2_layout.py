import unittest
import numpy as np
from .sam2_layout import detector_boxes


class LayoutTests(unittest.TestCase):
    def test_fragmented_empty_and_edge_slots_keep_exact_boxes(self):
        x = np.zeros((13, 17), np.uint16)
        x[0, 0] = x[-1, -1] = 1
        x[4:8, 6:9] = 3
        x[2, 16] = 4
        slots, boxes = detector_boxes(x, 5)
        self.assertEqual(slots, [0, 2, 3])
        np.testing.assert_array_equal(boxes, [[0, 0, 17, 13], [6, 4, 9, 8], [16, 2, 17, 3]])
        self.assertEqual(detector_boxes(np.zeros_like(x), 5)[1].shape, (0, 4))
        self.assertEqual(detector_boxes(np.zeros_like(x), 0)[0], [])

    def test_equivalence_to_existing_pixel_scan(self):
        rng = np.random.default_rng(20261029)
        for n in (1, 12, 128):
            x = rng.integers(0, n+1, (30, 40), dtype=np.uint16)
            slots, boxes = detector_boxes(x, n)
            expected = []
            for s in range(n):
                y, z = np.nonzero(x == s+1)
                if len(y):
                    expected.append([z.min(), y.min(), z.max()+1, y.max()+1])
            np.testing.assert_array_equal(boxes, expected)
            self.assertEqual(slots, list(range(n)))


if __name__ == '__main__':
    unittest.main()
