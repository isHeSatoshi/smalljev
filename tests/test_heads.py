"""v1 learned slot-head tests — pure torch, no backbone weights needed."""
import unittest


class TestSlotChoiceHead(unittest.TestCase):
    def test_uniform_at_init_for_any_options(self):
        # No letter bias by construction: zero-init head is exactly uniform,
        # unlike a pretrained LM head which prefers some letters (measured 0.68/0.32).
        from smalljev.heads import SlotChoiceHead
        import torch
        head = SlotChoiceHead(hidden_size=32, max_slots=8)
        h = torch.randn(4, 32)
        for k in (2, 3, 5, 8):
            p = head.probs(h, k)
            self.assertEqual(p.shape, (4, k))
            self.assertTrue(bool(((p - 1.0 / k).abs() < 1e-6).all()))

    def test_invalid_slots_exactly_zero(self):
        from smalljev.heads import SlotChoiceHead
        import torch
        head = SlotChoiceHead(hidden_size=16, max_slots=8)
        torch.nn.init.normal_(head.slot.weight)
        p = head.probs(torch.randn(2, 16), 3)
        self.assertEqual(p.shape, (2, 3))
        self.assertAlmostEqual(float(p.sum(1).min()), 1.0, places=5)

    def test_full_distribution_sums_to_one_over_all_slots(self):
        from smalljev.heads import SlotChoiceHead
        import torch
        head = SlotChoiceHead(hidden_size=16, max_slots=8)
        p = head.probs(torch.randn(2, 16), 8)
        self.assertAlmostEqual(float(p.detach().sum(1).max()), 1.0, places=5)

    def test_grad_flows_and_loss_decreases(self):
        from smalljev.heads import SlotChoiceHead
        import torch
        torch.manual_seed(0)
        head = SlotChoiceHead(hidden_size=16, max_slots=4)
        opt = torch.optim.SGD(head.parameters(), lr=0.5)
        h = torch.randn(32, 16)
        y = torch.tensor([i % 4 for i in range(32)])
        first, last = None, None
        for _ in range(50):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head.logits(h, 4), y)
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
            last = loss.item()
        self.assertLess(last, first * 0.5)

    def test_save_load_roundtrip(self):
        import os
        import tempfile
        from smalljev.heads import SlotChoiceHead
        import torch
        head = SlotChoiceHead(hidden_size=16, max_slots=8)
        torch.nn.init.normal_(head.slot.weight)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "heads.pt")
            head.save(path)
            head2 = SlotChoiceHead.load(path)
            self.assertEqual(head2.max_slots, 8)
            h = torch.randn(3, 16)
            self.assertTrue(bool((head.probs(h, 5) - head2.probs(h, 5)).abs().max() < 1e-6))

    def test_noul_head_neutral_at_init_and_learns(self):
        from smalljev.heads import BinaryNoulHead
        import torch
        torch.manual_seed(0)
        head = BinaryNoulHead(hidden_size=16)
        h = torch.randn(8, 16)
        self.assertTrue(bool(((head.prob(h) - 0.5).abs() < 1e-6).all()))
        opt = torch.optim.SGD(head.parameters(), lr=0.5)
        y = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy(head.prob(h), y)
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first * 0.7)

    def test_ordinal_head_uniform_at_init_and_learns(self):
        from smalljev.heads import OrdinalScoreHead
        import torch
        torch.manual_seed(0)
        head = OrdinalScoreHead(hidden_size=16, max_levels=8)
        h = torch.randn(6, 16)
        p = head.probs(h, 5)
        self.assertEqual(p.shape, (6, 5))
        self.assertTrue(bool(((p - 0.2).abs() < 1e-6).all()))
        self.assertAlmostEqual(float(p.sum(1).min()), 1.0, places=5)
        opt = torch.optim.SGD(head.parameters(), lr=0.5)
        y = torch.tensor([4, 4, 4, 0, 0, 0])
        first = None
        for _ in range(80):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head.logits(h, 5), y)
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first * 0.5)

    def test_ordinal_expected_value(self):
        from smalljev.heads import OrdinalScoreHead
        import torch
        head = OrdinalScoreHead(hidden_size=8, max_levels=8)
        with torch.no_grad():
            head.level.weight.zero_()
            head.level.bias.copy_(torch.tensor([-10., -10., 10., -10., -10., 0., 0., 0.]))
        h = torch.randn(2, 8)
        v = head.expected(h, 5, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertTrue(bool(((v - 0.5).abs() < 1e-3).all()))

    def test_bundle_roundtrip(self):
        import os
        import tempfile
        from smalljev.heads import HeadsBundle
        import torch
        b = HeadsBundle(hidden_size=16, max_slots=8, max_levels=6)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bundle.pt")
            b.save(path)
            b2 = HeadsBundle.load(path)
            h = torch.randn(2, 16)
            self.assertTrue(bool((b.choice.probs(h, 3) - b2.choice.probs(h, 3)).abs().max() < 1e-9))
            self.assertTrue(bool((b.noul.prob(h) - b2.noul.prob(h)).abs().max() < 1e-9))
            self.assertTrue(bool((b.score.probs(h, 4) - b2.score.probs(h, 4)).abs().max() < 1e-9))


if __name__ == "__main__":
    unittest.main()
