"""CPU-only synthetic contract tests; no datasets or accuracy claims.

Run from the repository root with:
    /home/frank/semantic_livo_runtime/venv/bin/python -B -m unittest tools.semantic_memory.test_memory -v

The ledger is tested directly for storage/retrieval bounds; the public module
is exercised for validation, causality, visibility ordering, and training.
"""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantic_memory import (  # noqa: E402
    CausalSemanticMemory,
    EvidenceMemory,
    HistoryFusion,
    MemoryConfig,
    training_loss,
    visibility,
)


def config(**overrides):
    values = dict(feature_dim=4, classes=3, hidden_dim=6, capacity=32,
                  cell_size=1.0, radius=0.75, records_per_cell=3, neighbors=4,
                  max_age_s=2.0, min_cosine=0.2, chunk_size=2, max_points=256)
    values.update(overrides)
    return MemoryConfig(**values)


def observation(points):
    xyz = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
    features = torch.tensor([[1.0, 0.25, -0.5, 0.75]]).repeat(len(xyz), 1)
    logits = torch.tensor([[2.0, 0.0, -1.0]]).repeat(len(xyz), 1)
    return xyz, features, logits


def force_residual(fusion):
    """Make a usable history change the answer, independent of random weights."""
    with torch.no_grad():
        for parameter in fusion.parameters():
            parameter.zero_()
        fusion.value.bias.fill_(1.0)
        fusion.delta.weight[0].fill_(-1.0)
        fusion.delta.weight[1].fill_(1.0)
        fusion.gate[-1].bias.fill_(2.0)


class MemoryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(173)

    def arguments(self, points=((5.125, 0.0, 0.0),), **overrides):
        xyz, features, logits = observation(points)
        args = dict(xyz_sensor=xyz, features=features, base_logits=logits,
                    stamp=0.0, frame=0, transform=torch.eye(4), epoch=0)
        args.update(overrides)
        return args

    def call(self, model, points=((5.125, 0.0, 0.0),), **overrides):
        return model(**self.arguments(points, **overrides))

    def snapshot(self, model):
        memory = model.memory
        return dict(tensors={name: getattr(memory, name).clone()
                             for name in (*memory.fields, 'readable')},
                    clock=(memory.last_stamp, memory.last_frame, memory.epoch),
                    status=dict(model.last_status))

    def assert_unchanged(self, model, before):
        for name, expected in before['tensors'].items():
            actual = getattr(model.memory, name)
            self.assertEqual(actual.dtype, expected.dtype, name)
            self.assertTrue(torch.equal(actual, expected), f'{name} mutated')
        self.assertEqual((model.memory.last_stamp, model.memory.last_frame,
                          model.memory.epoch), before['clock'])
        self.assertEqual(model.last_status, before['status'])

    def assert_baseline(self, output, logits):
        for field in ('logits', 'candidate_logits', 'base_logits'):
            self.assertTrue(torch.equal(output[field], logits.float()), field)
        self.assertFalse(bool(output['has_history'].any()))
        self.assertTrue(torch.equal(output['gate'], torch.zeros(len(logits))))

    def assert_ledger_bounds(self, memory):
        cfg = memory.cfg
        self.assertLessEqual(len(memory.xyz), cfg.capacity)
        for name in (*memory.fields, 'readable'):
            self.assertEqual(len(getattr(memory, name)), len(memory.xyz), name)
        if len(memory.xyz):
            self.assertTrue(bool((memory.key[1:] >= memory.key[:-1]).all()))
            _, counts = memory.key.unique(return_counts=True)
            self.assertLessEqual(int(counts.max()), cfg.records_per_cell)
            pairs = torch.stack([memory.key, memory.frame], dim=1)
            self.assertEqual(len(pairs.unique(dim=0)), len(pairs),
                             'A frame must not cast repeated votes in one cell')


class ConfigurationTests(MemoryTestCase):
    def test_invalid_integer_and_float_configuration_is_rejected(self):
        integer_fields = ('feature_dim', 'classes', 'capacity', 'records_per_cell',
                          'neighbors', 'chunk_size', 'hidden_dim', 'max_points',
                          'ray_width', 'ray_height')
        for field in integer_fields:
            for bad in (0, -1, 1.5, True):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ValueError):
                        config(**{field: bad})
        for field in ('cell_size', 'max_age_s', 'radius', 'ray_tolerance', 'clearance'):
            for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ValueError):
                        config(**{field: bad})

    def test_inconsistent_capacity_radius_classes_and_cosine_are_rejected(self):
        for changes in (dict(capacity=2, records_per_cell=3), dict(classes=1),
                        dict(radius=1.01), dict(min_cosine=-1.01),
                        dict(min_cosine=1.01), dict(min_cosine=math.nan)):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    config(**changes)


class CausalContractTests(MemoryTestCase):
    def test_empty_history_is_exact_baseline_across_chunks_and_dtypes(self):
        # Repeated points cross chunk boundaries; none may read its own frame.
        for dtype in (torch.float32, torch.float16, torch.float64):
            with self.subTest(dtype=dtype):
                model = CausalSemanticMemory(config())
                force_residual(model.fusion)
                args = self.arguments([(5.125, 0.0, 0.0)] * 7)
                args['features'] = args['features'].to(dtype)
                args['base_logits'] = torch.randn(7, 3).to(dtype)
                output = model(**args)
                self.assert_baseline(output, args['base_logits'])
                self.assertEqual(model.last_status['history_points'], 0)
                self.assertEqual(len(model.memory.xyz), 1)

    def test_empty_scan_has_correct_shapes_and_advances_causal_clock(self):
        model = CausalSemanticMemory(config())
        output = self.call(model, [], stamp=3.0, frame=7)
        self.assert_baseline(output, torch.empty(0, 3))
        self.assertEqual(output['logits'].shape, (0, 3))
        self.assertEqual(output['gate'].shape, (0,))
        self.assertEqual(output['has_history'].dtype, torch.bool)
        self.assertEqual((model.memory.last_stamp, model.memory.last_frame), (3.0, 7))
        self.assertEqual(model.last_status['records'], 0)

    def test_fusion_changes_predictions_but_writes_only_detached_raw_evidence(self):
        model = CausalSemanticMemory(config())
        force_residual(model.fusion)
        seed_features = torch.tensor([[1.0, 0.25, -0.5, 0.75]], requires_grad=True)
        seed_logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        self.call(model, features=seed_features, base_logits=seed_logits)
        old_features = model.memory.features.clone()
        old_probabilities = model.memory.probabilities.clone()
        raw_features = torch.tensor([[2.0, 0.5, -0.25, 1.0]], requires_grad=True)
        raw_logits = torch.tensor([[1.0, 0.5, -0.5]], requires_grad=True)
        output = self.call(model, stamp=0.25, frame=1,
                           features=raw_features, base_logits=raw_logits)
        self.assertTrue(bool(output['has_history'].all()))
        self.assertGreater(float(output['gate'].min().detach()), 0.5)
        self.assertFalse(torch.equal(output['logits'], raw_logits.detach()))
        new = model.memory.frame == 1
        old = model.memory.frame == 0
        self.assertEqual(int(new.sum()), 1)
        self.assertTrue(torch.equal(model.memory.features[new], raw_features.detach().half()))
        raw_probabilities = raw_logits.detach().float().softmax(-1).half()
        self.assertTrue(torch.equal(model.memory.probabilities[new], raw_probabilities))
        self.assertFalse(torch.equal(model.memory.probabilities[new],
                                     output['logits'].detach().softmax(-1).half()))
        self.assertTrue(torch.equal(model.memory.features[old], old_features))
        self.assertTrue(torch.equal(model.memory.probabilities[old], old_probabilities))
        for field in ('xyz', 'features', 'probabilities', 'stamp'):
            tensor = getattr(model.memory, field)
            self.assertFalse(tensor.requires_grad, field)
            self.assertIsNone(tensor.grad_fn, field)

    def test_shadow_output_is_baseline_with_real_history_and_raw_writes(self):
        model = CausalSemanticMemory(config())
        force_residual(model.fusion)
        self.call(model)
        args = self.arguments(stamp=0.1, frame=1, shadow=True)
        output = model(**args)
        self.assertTrue(bool(output['has_history'].all()))
        self.assertTrue(torch.equal(output['logits'], args['base_logits']))
        self.assertFalse(torch.equal(output['candidate_logits'], args['base_logits']))
        new = model.memory.frame == 1
        self.assertTrue(torch.equal(model.memory.probabilities[new],
                                    args['base_logits'].softmax(-1).half()))
        self.assertTrue(model.last_status['shadow'])

    def test_duplicate_or_backward_timestamp_or_frame_rejects_without_mutation(self):
        model = CausalSemanticMemory(config(max_age_s=20.0))
        self.call(model, stamp=10.0, frame=7, epoch=2)
        before = self.snapshot(model)
        for stamp, frame in ((10.0, 8), (9.0, 8), (11.0, 7), (11.0, 6), (10.0, 7)):
            with self.subTest(stamp=stamp, frame=frame):
                with self.assertRaisesRegex(ValueError, 'Duplicate/backward'):
                    self.call(model, stamp=stamp, frame=frame, epoch=2)
                self.assert_unchanged(model, before)
        output = self.call(model, stamp=11.0, frame=8, epoch=2)
        self.assertTrue(bool(output['has_history'].all()))

    def test_epoch_increase_resets_evidence_and_old_epoch_is_transactionally_rejected(self):
        model = CausalSemanticMemory(config())
        self.call(model, stamp=10.0, frame=10, epoch=2)
        args = self.arguments(stamp=11.0, frame=11, epoch=3)
        output = model(**args)
        self.assert_baseline(output, args['base_logits'])
        self.assertTrue(model.last_status['epoch_reset'])
        self.assertEqual(model.memory.frame.tolist(), [11])
        self.assertEqual(model.memory.epoch, 3)
        before = self.snapshot(model)
        with self.assertRaisesRegex(ValueError, 'Old coordinate epoch'):
            self.call(model, stamp=12.0, frame=12, epoch=2)
        self.assert_unchanged(model, before)
        output = self.call(model, stamp=12.0, frame=12, epoch=3)
        self.assertTrue(bool(output['has_history'].all()))
        self.assertFalse(model.last_status['epoch_reset'])

    def test_explicit_reset_allows_a_new_stream_with_restarted_clock(self):
        model = CausalSemanticMemory(config())
        self.call(model, stamp=100.0, frame=20, epoch=8)
        model.reset()
        self.assertEqual(len(model.memory.xyz), 0)
        self.assertIsNone(model.memory.epoch)
        self.assertIsNone(model.memory.last_stamp)
        self.assertIsNone(model.memory.last_frame)
        args = self.arguments()
        self.assert_baseline(model(**args), args['base_logits'])

    def test_nonfinite_inputs_and_timestamp_reject_before_reset_or_expiration(self):
        model = CausalSemanticMemory(config())
        self.call(model, stamp=1.0, frame=1, epoch=2)
        before = self.snapshot(model)
        for field in ('xyz_sensor', 'features', 'base_logits', 'transform'):
            for bad in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, bad=bad):
                    args = self.arguments(stamp=50.0, frame=2, epoch=3)
                    args[field].flatten()[0] = bad
                    with self.assertRaises(ValueError):
                        model(**args)
                    self.assert_unchanged(model, before)
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(stamp=bad):
                with self.assertRaises(ValueError):
                    self.call(model, stamp=bad, frame=2, epoch=3)
                self.assert_unchanged(model, before)

    def test_invalid_metadata_shapes_and_per_frame_point_limit_leave_state_unchanged(self):
        model = CausalSemanticMemory(config(max_points=3))
        self.call(model)
        before = self.snapshot(model)
        changes = [dict(frame=bad) for bad in (-1, 1.5, True)]
        changes += [dict(epoch=bad) for bad in (-1, 1.5, True)]
        changes += [dict(xyz_sensor=torch.zeros(1, 2)),
                    dict(features=torch.zeros(1, 5)),
                    dict(base_logits=torch.zeros(1, 4)),
                    dict(transform=torch.eye(3))]
        for change in changes:
            with self.subTest(change=change):
                args = self.arguments(stamp=10.0, frame=1, epoch=1)
                args.update(change)
                with self.assertRaises(ValueError):
                    model(**args)
                self.assert_unchanged(model, before)
        with self.assertRaises(ValueError):
            self.call(model, [(5.125, 0.0, 0.0)] * 4, stamp=10.0, frame=1, epoch=1)
        self.assert_unchanged(model, before)

    def test_rotation_and_homogeneous_transform_guards_are_atomic(self):
        model = CausalSemanticMemory(config())
        self.call(model)
        before = self.snapshot(model)
        transforms = []
        for row, column, value in ((0, 0, -1.0), (0, 0, 1.01),
                                   (0, 1, 0.01), (3, 0, 0.01), (3, 3, 2.0)):
            transform = torch.eye(4)
            transform[row, column] = value
            transforms.append(transform)
        for transform in transforms:
            with self.subTest(transform=transform):
                with self.assertRaises(ValueError):
                    self.call(model, stamp=10.0, frame=1, epoch=1, transform=transform)
                self.assert_unchanged(model, before)

    def test_world_key_range_and_reserved_boundary_margin_reject_before_mutation(self):
        model = CausalSemanticMemory(config())
        self.call(model)
        before = self.snapshot(model)
        limit = 1 << 20
        for x in (-limit - 1.0, -float(limit), float(limit - 1), float(limit)):
            with self.subTest(sensor_x=x):
                with self.assertRaisesRegex(ValueError, 'key range'):
                    self.call(model, [[x, 0.0, 0.0]], stamp=10.0, frame=1, epoch=1)
                self.assert_unchanged(model, before)
        transform = torch.eye(4)
        transform[0, 3] = float(limit)
        with self.assertRaisesRegex(ValueError, 'key range'):
            self.call(model, stamp=10.0, frame=1, epoch=1, transform=transform)
        self.assert_unchanged(model, before)

    def test_nearest_supported_boundary_cells_can_be_written_and_queried(self):
        limit = 1 << 20
        for x in (-limit + 1.25, limit - 1.75):
            with self.subTest(x=x):
                model = CausalSemanticMemory(config())
                self.call(model, [[x, 0.0, 0.0]])
                output = self.call(model, [[x, 0.0, 0.0]], stamp=0.1, frame=1)
                self.assertTrue(bool(output['has_history'].all()))

    def assert_finite_cast_or_atomic_rejection(self, **changes):
        # Either reject before mutation or represent safely. Input finiteness
        # alone does not imply finiteness after a storage/prediction downcast.
        model = CausalSemanticMemory(config())
        self.call(model)
        before = self.snapshot(model)
        args = self.arguments(stamp=10.0, frame=1, epoch=1)
        args.update(changes)
        try:
            output = model(**args)
        except ValueError:
            self.assert_unchanged(model, before)
            return
        for field in ('logits', 'candidate_logits', 'base_logits', 'gate'):
            self.assertTrue(bool(torch.isfinite(output[field]).all()), field)
        for field in ('xyz', 'features', 'probabilities'):
            self.assertTrue(bool(torch.isfinite(getattr(model.memory, field)).all()),
                            f'Nonfinite memory {field} from finite input')

    def test_finite_features_cannot_overflow_half_precision_storage(self):
        self.assert_finite_cast_or_atomic_rejection(features=torch.full((1, 4), 70000.0))

    def test_finite_float64_logits_cannot_overflow_float32_prediction(self):
        self.assert_finite_cast_or_atomic_rejection(
            base_logits=torch.full((1, 3), 1e100, dtype=torch.float64))

    def test_sensor_pose_is_used_for_both_lookup_and_visibility(self):
        model = CausalSemanticMemory(config())
        self.call(model)
        transform = torch.tensor([[0.0, -1.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0, 2.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        output = self.call(model, [[-2.0, -4.125, 0.0]], stamp=0.1, frame=1,
                           transform=transform)
        self.assertTrue(bool(output['has_history'].all()))
        self.assertEqual(model.last_status['revoked_free_space'], 0)
        self.assertEqual(model.last_status['occluded'], 0)
        self.assertTrue(torch.equal(model.memory.xyz, torch.tensor([[5.125, 0.0, 0.0]] * 2)))


class LedgerBoundTests(MemoryTestCase):
    def test_dense_frame_adds_one_vote_per_cell_and_retains_only_newest_cell_records(self):
        cfg = config(capacity=12, records_per_cell=3)
        memory = EvidenceMemory(cfg, 'cpu')
        points = [[5.125, 0.0, 0.0], [7.125, 0.0, 0.0], [9.125, 0.0, 0.0]]
        xyz, features, logits = observation(torch.tensor(points).repeat_interleave(40, dim=0))
        for frame in range(6):
            memory.write_raw(xyz, features + frame, logits.softmax(-1), float(frame), frame)
            self.assert_ledger_bounds(memory)
            self.assertEqual(len(memory.xyz), 3 * min(frame + 1, cfg.records_per_cell))
            self.assertEqual(set(memory.frame.tolist()),
                             set(range(max(0, frame - cfg.records_per_cell + 1), frame + 1)))
            for stored_frame in memory.frame.unique():
                selected = memory.frame == stored_frame
                expected = features[:1] + int(stored_frame)
                self.assertTrue(torch.equal(memory.features[selected], expected.half().repeat(3, 1)))

    def test_capacity_is_enforced_even_when_one_frame_exceeds_the_entire_budget(self):
        cfg = config(capacity=7, records_per_cell=2)
        memory = EvidenceMemory(cfg, 'cpu')
        for frame in range(4):
            points = [[5.125 + 2 * i, frame * 2.0, 0.0] for i in range(31)]
            xyz, features, logits = observation(points)
            memory.write_raw(xyz, features, logits.softmax(-1), float(frame), frame)
            self.assert_ledger_bounds(memory)
            self.assertEqual(len(memory.xyz), cfg.capacity)
            self.assertEqual(set(memory.frame.tolist()), {frame})

    def test_new_cell_replaces_old_evidence_in_a_full_ledger(self):
        memory = EvidenceMemory(config(capacity=5, records_per_cell=2), 'cpu')
        xyz, features, logits = observation([[5.125 + i * 2, 0.0, 0.0] for i in range(5)])
        memory.write_raw(xyz, features, logits.softmax(-1), 0.0, 0)
        xyz, features, logits = observation([[100.125, 0.0, 0.0]])
        memory.write_raw(xyz, features, logits.softmax(-1), 1.0, 1)
        self.assert_ledger_bounds(memory)
        self.assertEqual(len(memory.xyz), 5)
        self.assertEqual(int((memory.frame == 1).sum()), 1)
        self.assertTrue(torch.equal(memory.xyz[memory.frame == 1], xyz))

    def test_retrieval_requires_strictly_past_timestamp_radius_and_cosine(self):
        memory = EvidenceMemory(config(radius=0.5, records_per_cell=4, neighbors=8), 'cpu')
        for frame, (x, sign) in enumerate(((5.125, 1), (5.25, -1), (5.875, 1), (5.125, 1))):
            xyz, _, logits = observation([[x, 0.0, 0.0]])
            features = torch.tensor([[float(sign), 0.0, 0.0, 0.0]])
            memory.write_raw(xyz, features, logits.softmax(-1), float(frame), frame)
        xyz, _, _ = observation([[5.125, 0.0, 0.0]])
        query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        candidates = memory.retrieve(xyz, query, stamp=3.0)
        valid = candidates['valid']
        self.assertEqual(int(valid.sum()), 1)
        self.assertEqual(candidates['frame'][valid].tolist(), [0])
        self.assertEqual(candidates['age'][valid].tolist(), [3.0])
        self.assertTrue(bool((candidates['delta'][valid].norm(dim=-1) <= 0.5).all()))

    def test_candidate_shortlist_stays_bounded_when_distant_memory_grows(self):
        for neighbors in (5, 100):
            with self.subTest(neighbors=neighbors):
                cfg = config(capacity=2048, records_per_cell=2, neighbors=neighbors, radius=1.0)
                memory = EvidenceMemory(cfg, 'cpu')
                local = torch.cartesian_prod(*(torch.tensor([9.875, 10.125]) for _ in range(3)))
                xyz, features, logits = observation(local)
                for frame in range(2):
                    memory.write_raw(xyz, features, logits.softmax(-1), float(frame), frame)
                query, query_features, _ = observation([[10.0, 10.0, 10.0]] * 17)
                small = memory.retrieve(query, query_features, stamp=3.0)
                distant = [[100.125 + 2 * i, 0.0, 0.0] for i in range(1500)]
                xyz, features, logits = observation(distant)
                memory.write_raw(xyz, features, logits.softmax(-1), 2.0, 2)
                large = memory.retrieve(query, query_features, stamp=3.0)
                expected_k = min(neighbors, 27 * cfg.records_per_cell)
                self.assertGreater(len(memory.xyz), 1000)
                for candidates in (small, large):
                    self.assertEqual(candidates['valid'].shape, (17, expected_k))
                    self.assertEqual(candidates['features'].shape, (17, expected_k, cfg.feature_dim))
                    self.assertLessEqual(int(candidates['valid'].sum(1).max()), cfg.neighbors)
                    self.assertTrue(bool((candidates['valid'].sum(1) > 0).all()))
                # Compare accepted evidence only: padding indices are unspecified.
                for field in ('features', 'probabilities', 'delta', 'age', 'frame'):
                    self.assertTrue(torch.equal(small[field][small['valid']],
                                                large[field][large['valid']]), field)

    def test_public_forward_bounds_each_fusion_batch_and_candidate_count(self):
        cfg = config(chunk_size=3, neighbors=5, records_per_cell=2)
        model = CausalSemanticMemory(cfg)
        self.call(model)
        with patch.object(model.fusion, 'forward', wraps=model.fusion.forward) as forward:
            output = self.call(model, [(5.125, 0.0, 0.0)] * 11, stamp=0.1, frame=1)
        self.assertEqual(forward.call_count, 4)
        self.assertEqual([len(call.args[0]) for call in forward.call_args_list], [3, 3, 3, 2])
        for call in forward.call_args_list:
            candidates = call.args[2]
            self.assertLessEqual(candidates['valid'].shape[1], cfg.neighbors)
            self.assertLessEqual(candidates['valid'].shape[1], 27 * cfg.records_per_cell)
            self.assertTrue(bool((candidates['frame'][candidates['valid']] < 1).all()))
        self.assertTrue(bool(output['has_history'].all()))


class VisibilityAndLifetimeTests(MemoryTestCase):
    def test_visibility_distinguishes_surface_occluded_free_and_unobserved(self):
        memory = torch.tensor([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                               [0.0, 5.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = visibility(memory, torch.tensor([[5.0, 0.0, 0.0]]), config())
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(result.tolist(), [1, 2, 3, 0, 0, 0])

    def test_nearest_return_controls_a_ray_regardless_of_input_order(self):
        memory = torch.tensor([[10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        scan = torch.tensor([[10.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        for rays in (scan, scan.flip(0)):
            with self.subTest(rays=rays.tolist()):
                self.assertEqual(visibility(memory, rays, config()).tolist(), [2, 1, 3])

    def test_missing_returns_and_angular_bin_collision_are_not_free_space(self):
        memory = torch.tensor([[5.0, 0.0, 0.0]])
        self.assertEqual(visibility(memory, torch.empty(0, 3), config()).tolist(), [0])
        self.assertEqual(visibility(torch.empty(0, 3), memory, config()).shape, (0,))
        cfg = config(ray_width=1, ray_height=1, ray_tolerance=0.01)
        off_ray = torch.tensor([[5.0 * math.cos(0.1), 5.0 * math.sin(0.1), 0.0]])
        self.assertEqual(visibility(off_ray, memory * 2, cfg).tolist(), [0])
        self.assertEqual(visibility(memory, torch.zeros(1, 3), cfg).tolist(), [0])

    def test_clearance_is_inclusive_at_the_surface_boundary(self):
        memory = torch.tensor([[5.0, 0.0, 0.0]])
        cfg = config(clearance=0.5)
        for depth, expected in ((4.5, 1), (5.5, 1), (4.375, 2), (5.625, 3)):
            with self.subTest(depth=depth):
                scan = torch.tensor([[depth, 0.0, 0.0]])
                self.assertEqual(visibility(memory, scan, cfg).tolist(), [expected])

    def test_free_space_retires_evidence_before_it_can_change_current_prediction(self):
        # Both points fit the retrieval radius and have identical features.
        model = CausalSemanticMemory(config(radius=1.0))
        force_residual(model.fusion)
        self.call(model, [[5.125, 0.0, 0.0]])
        args = self.arguments([[5.625, 0.0, 0.0]], stamp=0.1, frame=1)
        output = model(**args)
        self.assert_baseline(output, args['base_logits'])
        self.assertEqual(model.last_status['revoked_free_space'], 1)
        self.assertEqual(model.memory.frame.tolist(), [1])

    def test_occluded_evidence_is_kept_but_not_read_and_can_be_seen_again(self):
        model = CausalSemanticMemory(config(radius=1.0))
        force_residual(model.fusion)
        self.call(model, [[5.625, 0.0, 0.0]])
        args = self.arguments([[5.125, 0.0, 0.0]], stamp=0.1, frame=1)
        self.assert_baseline(model(**args), args['base_logits'])
        self.assertEqual(model.last_status['occluded'], 1)
        self.assertEqual(model.last_status['revoked_free_space'], 0)
        self.assertEqual(set(model.memory.frame.tolist()), {0, 1})
        output = self.call(model, [[5.625, 0.0, 0.0]], stamp=0.2, frame=2)
        self.assertTrue(bool(output['has_history'].all()))
        self.assertEqual(model.last_status['revoked_free_space'], 1)
        self.assertIn(0, model.memory.frame.tolist())
        self.assertNotIn(1, model.memory.frame.tolist())

    def test_unobserved_evidence_is_retained_and_remains_available(self):
        model = CausalSemanticMemory(config())
        self.call(model)
        output = self.call(model, [[5.125, 0.25, 0.0]], stamp=0.1, frame=1)
        self.assertTrue(bool(output['has_history'].all()))
        self.assertEqual(model.last_status['revoked_free_space'], 0)
        self.assertEqual(model.last_status['occluded'], 0)
        self.assertIn(0, model.memory.frame.tolist())

    def test_ttl_keeps_exact_limit_but_expires_before_read_just_after_limit(self):
        for age, kept in ((0.5, True), (1.0, True), (1.0001, False)):
            with self.subTest(age=age):
                model = CausalSemanticMemory(config(max_age_s=1.0))
                force_residual(model.fusion)
                self.call(model)
                args = self.arguments(stamp=age, frame=1)
                output = model(**args)
                self.assertEqual(bool(output['has_history'].item()), kept)
                self.assertEqual(model.last_status['expired'], int(not kept))
                self.assertEqual(0 in model.memory.frame.tolist(), kept)
                if not kept:
                    self.assert_baseline(output, args['base_logits'])

    def test_empty_scan_preserves_unexpired_evidence_but_still_runs_ttl(self):
        model = CausalSemanticMemory(config(max_age_s=1.0))
        self.call(model)
        self.call(model, [], stamp=0.5, frame=1)
        self.assertEqual(model.memory.frame.tolist(), [0])
        self.assertEqual(model.last_status['revoked_free_space'], 0)
        self.assertEqual(model.last_status['expired'], 0)
        self.call(model, [], stamp=1.25, frame=2)
        self.assertEqual(len(model.memory.xyz), 0)
        self.assertEqual(model.last_status['expired'], 1)


class FusionAndTrainingTests(MemoryTestCase):
    def test_eval_rejected_history_returns_exact_baseline_but_can_train(self):
        fusion = HistoryFusion(config())
        force_residual(fusion)
        with torch.no_grad():
            fusion.gate[-1].weight.zero_()
            fusion.gate[-1].bias.fill_(-2.)
        features, logits = torch.randn(3, 4), torch.randn(3, 3)
        candidates = self.candidates()
        candidates['valid'][:] = True
        fusion.eval()
        rejected = fusion(features, logits, candidates)
        self.assertTrue(torch.equal(rejected['logits'], logits))
        self.assertFalse(torch.equal(rejected['candidate_logits'], logits))
        fusion.train()
        soft = fusion(features, logits, candidates)
        self.assertFalse(torch.equal(soft['logits'], logits))
        fusion.eval()
        with torch.no_grad():
            fusion.gate[-1].bias.fill_(2.)
        accepted = fusion(features, logits, candidates)
        self.assertFalse(torch.equal(accepted['logits'], logits))

    def candidates(self, rows=3, neighbors=4):
        return dict(features=torch.randn(rows, neighbors, 4),
                    probabilities=torch.randn(rows, neighbors, 3).softmax(-1),
                    delta=torch.zeros(rows, neighbors, 3),
                    age=torch.ones(rows, neighbors) * 0.1,
                    valid=torch.zeros(rows, neighbors, dtype=torch.bool),
                    frame=torch.zeros(rows, neighbors, dtype=torch.long))

    def test_no_history_skips_attention_and_gate_and_is_exact_with_biased_layers(self):
        fusion = HistoryFusion(config())
        force_residual(fusion)
        features = torch.randn(3, 4)
        logits = torch.randn(3, 3, requires_grad=True)
        candidates = self.candidates()
        with patch.object(fusion.query, 'forward', wraps=fusion.query.forward) as query, \
             patch.object(fusion.gate, 'forward', wraps=fusion.gate.forward) as gate:
            output = fusion(features, logits, candidates)
        self.assert_baseline(output, logits)
        query.assert_not_called()
        gate.assert_not_called()
        output['logits'].sum().backward()
        self.assertTrue(torch.equal(logits.grad, torch.ones_like(logits)))

    def test_mixed_history_only_changes_supported_rows_and_preserves_order(self):
        fusion = HistoryFusion(config())
        force_residual(fusion)
        features = torch.randn(4, 4)
        logits = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        candidates = self.candidates(rows=4)
        candidates['valid'][0, 0] = True
        candidates['valid'][2, :2] = True
        output = fusion(features, logits, candidates)
        self.assertEqual(output['has_history'].tolist(), [True, False, True, False])
        inactive = ~output['has_history']
        self.assertTrue(torch.equal(output['logits'][inactive], logits[inactive]))
        self.assertTrue(torch.equal(output['candidate_logits'][inactive], logits[inactive]))
        self.assertTrue(torch.equal(output['gate'][inactive], torch.zeros(2)))
        self.assertTrue(bool((output['gate'][~inactive] > 0).all()))
        self.assertTrue(bool((output['gate'][~inactive] < 1).all()))
        self.assertFalse(torch.equal(output['logits'][~inactive], logits[~inactive]))
        self.assertTrue(bool(torch.isfinite(output['logits']).all()))

    def test_training_gate_uses_correctness_improvement_even_for_identical_high_confidence(self):
        # Equal confidence, opposite correctness. History helps row 0 and hurts
        # row 1; a confidence-only selector cannot distinguish these targets.
        base = torch.tensor([[0.0, 12.0, 0.0], [0.0, 12.0, 0.0]], requires_grad=True)
        candidate = torch.tensor([[12.0, 0.0, 0.0], [12.0, 0.0, 0.0]], requires_grad=True)
        gate = torch.tensor([0.25, 0.75], requires_grad=True)
        labels = torch.tensor([0, 1])
        confidence, prediction = base.detach().softmax(-1).max(-1)
        self.assertGreater(float(confidence.min()), 0.99)
        self.assertEqual(confidence[0].item(), confidence[1].item())
        self.assertNotEqual(prediction[0].item(), labels[0].item())
        self.assertEqual(prediction[1].item(), labels[1].item())
        output = dict(base_logits=base, candidate_logits=candidate, gate=gate,
                      logits=base + gate[:, None] * (candidate - base),
                      has_history=torch.ones(2, dtype=torch.bool))
        loss, metrics = training_loss(output, labels)
        expected_gate_loss = F.binary_cross_entropy(gate, torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(metrics['gate'], float(expected_gate_loss.detach()), places=6)
        self.assertEqual(metrics['history_supervised'], 2)
        loss.backward()
        self.assertLess(float(gate.grad[0]), 0.0, 'Helpful history must open the gate')
        self.assertGreater(float(gate.grad[1]), 0.0, 'Harmful history must close the gate')

    def test_loss_ignores_negative_labels_and_trains_gate_only_with_history(self):
        base = torch.tensor([[0.0, 3.0, 0.0]] * 3, requires_grad=True)
        candidate = torch.tensor([[3.0, 0.0, 0.0]] * 3, requires_grad=True)
        logits = torch.tensor([[1.0, 0.0, 0.0]] * 3, requires_grad=True)
        gate = torch.tensor([0.25, 0.5, 0.75], requires_grad=True)
        labels = torch.tensor([0, -1, 2])
        output = dict(logits=logits, candidate_logits=candidate, base_logits=base,
                      gate=gate, has_history=torch.tensor([True, True, False]))
        loss, metrics = training_loss(output, labels)
        expected = (F.cross_entropy(logits[[0, 2]], labels[[0, 2]])
                    + 0.2 * -gate[0].log()
                    + 0.2 * F.cross_entropy(candidate[:1], labels[:1]))
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-7)
        self.assertEqual(metrics['supervised'], 2)
        self.assertEqual(metrics['history_supervised'], 1)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(3)))
        self.assertTrue(torch.equal(candidate.grad[1:], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(gate.grad[1:], torch.zeros(2)))
        self.assertIsNone(base.grad, 'Detached selector targets must not backpropagate into baseline')

    def test_no_history_loss_is_cross_entropy_and_still_supports_backward(self):
        model = CausalSemanticMemory(config())
        args = self.arguments([(5.125, 0.0, 0.0)] * 3)
        args['base_logits'].requires_grad_()
        output = model(**args)
        labels = torch.tensor([0, 1, 2])
        loss, metrics = training_loss(output, labels)
        torch.testing.assert_close(loss, F.cross_entropy(args['base_logits'], labels), rtol=0, atol=0)
        self.assertEqual(metrics['gate'], 0.0)
        self.assertEqual(metrics['auxiliary'], 0.0)
        self.assertEqual(metrics['history_supervised'], 0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(args['base_logits'].grad).all()))

    def test_training_loss_rejects_frames_without_supervised_labels(self):
        fusion = HistoryFusion(config())
        for labels in (torch.tensor([-1, -2]), torch.empty(0, dtype=torch.long)):
            with self.subTest(labels=labels.tolist()):
                output = fusion(torch.zeros(len(labels), 4), torch.zeros(len(labels), 3),
                                self.candidates(rows=len(labels)))
                with self.assertRaisesRegex(ValueError, 'no supervised labels'):
                    training_loss(output, labels)

    def test_auxiliary_loss_reaches_residual_even_with_a_closed_gate(self):
        model = CausalSemanticMemory(config())
        with torch.no_grad():
            model.fusion.gate[-1].weight.zero_()
            model.fusion.gate[-1].bias.fill_(-30.0)
        self.call(model)
        output = self.call(model, stamp=0.1, frame=1)
        self.assertLess(float(output['gate'].max().detach()), 1e-10)
        loss, metrics = training_loss(output, torch.tensor([1]))
        loss.backward()
        self.assertGreater(metrics['auxiliary'], 0.0)
        self.assertGreater(float(model.fusion.delta.weight.grad.norm()), 1e-4)

    def test_short_synthetic_training_updates_gate_and_residual_without_recurrent_gradients(self):
        cfg = config(chunk_size=3, min_cosine=-1.0)
        model = CausalSemanticMemory(cfg)
        optimizer = torch.optim.Adam(model.fusion.parameters(), lr=0.01)
        points = [[4.875, 0.125, 0.0], [5.125, 0.125, 0.0],
                  [4.875, 0.25, 0.0], [5.125, 0.25, 0.0]]
        xyz, _, _ = observation(points)
        history_features = torch.randn(4, 4, requires_grad=True)
        history_logits = torch.randn(4, 3, requires_grad=True)
        current_features = torch.randn(4, 4)
        base_logits = torch.tensor([[2.0, 0.0, -1.0]] * 4)
        labels = torch.tensor([1, 2, 1, -1])
        initial_gate = model.fusion.gate[-1].bias.detach().clone()
        initial_delta = model.fusion.delta.weight.detach().clone()
        for step in range(5):
            model.reset()
            optimizer.zero_grad(set_to_none=True)
            model(xyz, history_features, history_logits, stamp=0.0, frame=0,
                  transform=torch.eye(4))
            features = current_features.clone().requires_grad_()
            logits = base_logits.clone().requires_grad_()
            output = model(xyz, features, logits, stamp=0.1, frame=1, transform=torch.eye(4))
            self.assertTrue(bool(output['has_history'].all()), step)
            loss, metrics = training_loss(output, labels)
            self.assertTrue(bool(torch.isfinite(loss)), step)
            self.assertEqual(metrics['supervised'], 3)
            self.assertEqual(metrics['history_supervised'], 3)
            self.assertTrue(all(math.isfinite(metrics[key])
                                for key in ('segmentation', 'gate', 'auxiliary')))
            loss.backward()
            for name, parameter in model.fusion.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            for parameter in (model.fusion.delta.weight, model.fusion.gate[-1].bias):
                self.assertGreater(float(parameter.grad.norm()), 0.0)
            self.assertTrue(bool(torch.isfinite(features.grad).all()))
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
            self.assertIsNone(history_features.grad)
            self.assertIsNone(history_logits.grad)
            self.assertFalse(model.memory.features.requires_grad)
            self.assertFalse(model.memory.probabilities.requires_grad)
            optimizer.step()
        self.assertFalse(torch.equal(model.fusion.gate[-1].bias.detach(), initial_gate))
        self.assertFalse(torch.equal(model.fusion.delta.weight.detach(), initial_delta))


if __name__ == '__main__':
    unittest.main()
