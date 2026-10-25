"""Equation/gradient tests for the explicit native PyTorch compatibility port."""
import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from .runtime import attention, segment_csr
from .train import inventory


class RuntimeTests(unittest.TestCase):
    def test_training_excludes_validation_test_and_incomplete_full_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            for sequences in [['08'], ['11'], ['04', '04']]:
                with self.assertRaises(ValueError):
                    inventory(tmp, sequences)
            with self.assertRaises(ValueError):
                inventory(tmp, ['04'], full=True)
            base = Path(tmp) / 'sequences/04'
            (base / 'velodyne').mkdir(parents=True)
            (base / 'labels').mkdir()
            np.zeros((20, 4), dtype='<f4').tofile(base / 'velodyne/000000.bin')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                inventory(tmp, ['04'])
            np.full(20, 10, dtype='<u4').tofile(base / 'labels/000000.label')
            self.assertEqual(len(inventory(tmp, ['04'])), 1)

    def test_attention_matches_independent_dense_equation_and_gradients(self):
        torch.manual_seed(16)
        qkv = torch.randn(16, 3, 2, 8, dtype=torch.float64, requires_grad=True)
        boundaries = torch.tensor([0, 5, 10, 16], dtype=torch.int32)
        actual = attention(qkv, boundaries, 6, softmax_scale=.3)
        expected = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            q, k, v = qkv[lo:hi].permute(1, 2, 0, 3).unbind(0)
            expected.append(((q @ k.transpose(-2, -1) * .3).softmax(-1) @ v).transpose(0, 1))
        expected = torch.cat(expected)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.square().sum(), qkv, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.square().sum(), qkv)[0]
        torch.testing.assert_close(ga, ge)

    def test_segment_reductions_match_dense_segments(self):
        torch.manual_seed(19)
        source = torch.randn(14, 4, dtype=torch.float64, requires_grad=True)
        boundaries = torch.tensor([0, 3, 8, 14])
        for reduction in ['sum', 'mean', 'min', 'max']:
            got = segment_csr(source, boundaries, reduce=reduction)
            expected = []
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                part = source[lo:hi]
                value = getattr(part, reduction)(dim=0)
                expected.append(value if reduction in ('sum', 'mean') else value.values)
            expected = torch.stack(expected)
            torch.testing.assert_close(got, expected)
            ga = torch.autograd.grad(got.sum(), source, retain_graph=True)[0]
            ge = torch.autograd.grad(expected.sum(), source, retain_graph=True)[0]
            torch.testing.assert_close(ga, ge)


if __name__ == '__main__':
    unittest.main()
