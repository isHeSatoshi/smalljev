"""Readout unit tests — pure torch, no weights."""
import unittest

import torch

from smalljev.backbones import BACKBONES, cls_readout, last_readout, mean_readout


class TestReadouts(unittest.TestCase):
    def test_cls_takes_first(self):
        h = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        out = cls_readout(h)
        self.assertEqual(float((out - h[:, 0, :]).abs().max()), 0.0)

    def test_mean_ignores_pads(self):
        h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        out = mean_readout(h, mask)
        want = torch.tensor([[2.0, 2.0]])
        self.assertLess(float((out - want).abs().max()), 1e-6)

    def test_last_takes_last_nonpad(self):
        h = torch.zeros(2, 3, 2)
        h[0, 0, 0] = 1.0
        h[0, 1, 0] = 2.0
        h[1, 0, 0] = 5.0
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        out = last_readout(h, mask, "cpu")
        want = torch.zeros(2, 2)
        want[0, 0] = 2.0
        want[1, 0] = 5.0
        self.assertEqual(float((out - want).abs().max()), 0.0)

    def test_registry_covers_tiers(self):
        for key in ("minicpm5", "bidirlm", "ettin-1b", "ettin-from-dec-1b",
                    "modernbert-base", "modernbert-large", "f2llm-4b",
                    "harrier-0.6b"):
            self.assertIn(key, BACKBONES)
            self.assertIn(BACKBONES[key]["readout"], ("cls", "mean", "last"))
            self.assertIn(BACKBONES[key]["license"], ("Apache-2.0", "MIT"))


if __name__ == "__main__":
    unittest.main()
