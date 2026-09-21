"""Per-option semantic scoring tests — pure torch, no weights.

The anti-position-bias property: a SHARED scorer applied to per-option
representations must permute exactly when options permute (unlike slot heads,
whose slot i is positional).
"""
import unittest


class TestFindSpans(unittest.TestCase):
    def test_sequential_spans_found(self):
        from smalljev.heads import find_spans
        full = [1, 2, 10, 11, 12, 20, 21, 30]
        spans = find_spans(full, [[10, 11, 12], [20, 21]])
        self.assertEqual(spans, [(2, 5), (5, 7)])

    def test_missing_option_falls_back(self):
        from smalljev.heads import find_spans
        full = [1, 2, 3]
        spans = find_spans(full, [[1, 2], [99]])
        self.assertEqual(spans[0], (0, 2))
        self.assertEqual(spans[1][0], spans[1][1])  # empty span = fallback marker


class TestOptionScorer(unittest.TestCase):
    def test_uniform_at_init(self):
        from smalljev.heads import OptionScorerHead
        import torch
        head = OptionScorerHead(hidden_size=16)
        H = torch.randn(3, 4, 16)  # batch, k, hidden
        p = head.probs(H)
        self.assertEqual(p.shape, (3, 4))
        self.assertTrue(bool(((p - 0.25).abs() < 1e-6).all()))

    def test_permutation_equivariance(self):
        from smalljev.heads import OptionScorerHead
        import torch
        torch.manual_seed(0)
        head = OptionScorerHead(hidden_size=16)
        torch.nn.init.normal_(head.scorer.weight)
        H = torch.randn(2, 3, 16)
        p1 = head.probs(H)
        p2 = head.probs(H[:, [2, 0, 1], :])
        self.assertTrue(bool((((p1[:, [2, 0, 1]] - p2).abs().max() < 1e-5))))

    def test_learns_separation(self):
        from smalljev.heads import OptionScorerHead
        import torch
        torch.manual_seed(0)
        head = OptionScorerHead(hidden_size=8)
        opt = torch.optim.SGD(head.parameters(), lr=0.5)
        H = torch.randn(24, 2, 8)
        H[:, 0, :] += 2.0
        y = torch.zeros(24, dtype=torch.long)
        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head.logits(H), y)
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first * 0.5)

    def test_save_load(self):
        import os
        import tempfile
        from smalljev.heads import OptionScorerHead
        import torch
        head = OptionScorerHead(hidden_size=12)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "opt.pt")
            head.save(p)
            h2 = OptionScorerHead.load(p)
            H = torch.randn(2, 5, 12)
            self.assertTrue(bool((head.probs(H) - h2.probs(H)).abs().max() < 1e-9))


if __name__ == "__main__":
    unittest.main()
