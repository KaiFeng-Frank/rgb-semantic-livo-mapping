"""Causal/fallback, moving-observation and repair-objective regression tests."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from semantic_memory import MemoryConfig
from semantic_repair import CurrentAnchoredRepair, RepairConfig, repair_loss


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        cfg = MemoryConfig(feature_dim=4, classes=3, cell_size=1.5, radius=1.5,
            records_per_cell=4, neighbors=4, min_cosine=.7, max_age_s=.75, capacity=32)
        repair = RepairConfig(max_queries=2, moving_classes=(1,))
        self.model = CurrentAnchoredRepair(cfg, repair)

    def call(self, xyz, frame, **options):
        xyz = torch.tensor(xyz, dtype=torch.float32).reshape(-1,3)
        n = len(xyz)
        features = torch.tensor([[1.,.5,-.5,1.]]).repeat(n,1)
        base = torch.tensor([[.2,.6,.1]]).repeat(n,1)
        anchor = torch.tensor([[.3,.7,.2]]).repeat(n,1)
        options.setdefault('features', features)
        options.setdefault('base_logits', base)
        options.setdefault('anchor_logits', anchor)
        return self.model(xyz, stamp=.5*frame, frame=frame, transform=torch.eye(4), **options)

    def test_missing_history_equals_learned_current_path_even_with_nonzero_head(self):
        with torch.no_grad():
            self.model.fusion.net[-1].bias.fill_(1)
        a = self.call([[5.,0,0]], 0)
        b = self.call([[5.,0,0]], 1, disable_history=True)
        self.model.reset()
        c = self.call([[5.,0,0]], 0, current_only=True)
        for out in (a,b):
            self.assertTrue(torch.equal(out['logits'], c['logits']))
        self.assertFalse(torch.equal(a['logits'], torch.tensor([[.2,.6,.1]])))

    def test_missing_history_still_trains_current_features(self):
        out = self.call([[5.,0,0]], 0)
        loss = repair_loss(out, torch.tensor([0]), torch.ones(3))
        loss.backward()
        self.assertGreater(self.model.fusion.net[-1].weight.grad.abs().sum().item(), 0)

    def test_moving_evidence_survives_vacated_position_and_wider_search(self):
        self.call([[5.,0,0]], 0)
        out = self.call([[6.,0,0]], 1)
        self.assertTrue(out['has_history'].item())
        self.assertEqual(self.model.last_status['revoked_free_space'], 0)
        self.model.reset()
        self.call([[5.,0,0]], 0)
        narrow = self.call([[6.,0,0]], 1, narrow_history=True)
        self.assertFalse(narrow['has_history'].item())

    def test_static_free_space_still_revokes_and_age_expires(self):
        self.call([[5.,0,0]], 0, base_logits=torch.tensor([[2.,0.,0.]]))
        self.call([[6.,0,0]], 1)
        self.assertEqual(self.model.last_status['revoked_free_space'], 1)
        out = self.call([[6.,0,0]], 3)
        self.assertFalse(out['has_history'].item())

    def test_class_representatives_and_raw_only_single_frame_writes(self):
        base = torch.tensor([[2.,0.,0.],[0.,2.,0.]])
        out = self.call([[5.,0,0],[5.1,0,0]], 0, base_logits=base)
        self.assertEqual(len(self.model.memory.xyz), 2)
        self.assertEqual(set(self.model.memory.probabilities.argmax(-1).tolist()), {0,1})
        self.call([[5.1,0,0]], 1)
        self.assertEqual(self.model.memory.frame.tolist(), [1])
        self.assertFalse(self.model.memory.features.requires_grad)
        self.assertTrue(torch.allclose(self.model.memory.probabilities.float(),
                                      torch.tensor([[.2,.6,.1]]).softmax(-1), atol=.001))

    def test_query_budget_initialization_and_same_train_eval_behavior(self):
        out = self.call([[5.+i,0,0] for i in range(5)], 0, current_only=True)
        self.assertEqual(len(out['query_indices']), 2)
        self.assertTrue(torch.equal(out['logits'], out['anchor_logits']))
        with torch.no_grad():
            self.model.fusion.net[-1].weight.normal_(0,.1)
        self.model.reset()
        a = self.call([[5.,0,0]], 0, current_only=True)
        self.model.reset(); self.model.eval()
        b = self.call([[5.,0,0]], 0, current_only=True)
        self.assertTrue(torch.equal(a['logits'], b['logits']))

    def test_duplicate_timestamp_does_not_mutate_memory(self):
        self.call([[5.,0,0]], 0)
        before = self.model.memory.xyz.clone()
        with self.assertRaises(ValueError):
            self.call([[6.,0,0]], 0)
        self.assertTrue(torch.equal(before, self.model.memory.xyz))

    def test_loss_prefers_repair_and_preservation_over_new_errors(self):
        anchor = torch.tensor([[2.,0.,0.],[0.,2.,0.]])
        good = torch.tensor([[2.,0.,0.],[2.,0.,0.]], requires_grad=True)
        bad = torch.tensor([[0.,2.,0.],[0.,2.,0.]], requires_grad=True)
        def output(x):
            return dict(anchor_logits=anchor, query_indices=torch.arange(2), selected_logits=x)
        labels = torch.tensor([0,0])
        a = repair_loss(output(good), labels, torch.ones(3))
        b = repair_loss(output(bad), labels, torch.ones(3))
        self.assertLess(a.item(), b.item())
        a.backward()
        self.assertTrue(torch.isfinite(good.grad).all())


if __name__ == '__main__':
    unittest.main()
