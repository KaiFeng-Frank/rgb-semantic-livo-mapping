import unittest

from .semantic_delivery import match_maps


class MapDeliveryTests(unittest.TestCase):
    def info(self, identity, ready):
        return dict(publication_id=identity, graph_stamp_ns=100, map_sha256='same',
                    map_points=3, ready_monotonic=ready)

    def cloud(self, received):
        return dict(stamp_ns=100, sha256='same', points=3, frame_id='map', received=received)

    def test_identical_maps_require_distinct_receptions(self):
        pairs = match_maps([self.info(1, 1.), self.info(2, 2.)], [self.cloud(3.)])
        self.assertIsNotNone(pairs[0][1])
        self.assertIsNone(pairs[1][1])

    def test_prior_reception_cannot_prove_new_snapshot(self):
        self.assertIsNone(match_maps([self.info(2, 2.)], [self.cloud(1.9)])[0][1])

    def test_two_distinct_receptions_are_matched_once_in_order(self):
        inputs = [self.cloud(3.), self.cloud(1.1)]
        pairs = match_maps([self.info(2, 2.), self.info(1, 1.)], inputs)
        self.assertEqual([c['received'] for _, c in pairs], [1.1, 3.])
        self.assertEqual(len(inputs), 2)

    def test_geometry_frame_stamp_hash_and_count_must_match(self):
        for key, value in [('frame_id', 'odom'), ('stamp_ns', 101), ('sha256', 'other'), ('points', 4)]:
            with self.subTest(field=key):
                cloud = dict(self.cloud(3.), **{key: value})
                self.assertIsNone(match_maps([self.info(1, 1.)], [cloud])[0][1])
