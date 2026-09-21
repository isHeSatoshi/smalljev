"""PPO-over-decisions math tests — pure torch, no weights.
Policy acts in decision space (choose slot from sequence embedding);
no token generation anywhere in the loop.
"""
import unittest


class TestPPOLoss(unittest.TestCase):
    def test_clipping_binds_positive_advantage(self):
        from smalljev.ppo import ppo_policy_loss
        import torch
        # ratio = 2.0, adv = +1: unclipped would be -2.0, clipped -1.2
        logp_new = torch.log(torch.tensor([0.8]))
        logp_old = torch.log(torch.tensor([0.4]))
        adv = torch.tensor([1.0])
        loss = ppo_policy_loss(logp_new, logp_old, adv, clip=0.2)
        self.assertAlmostEqual(loss.item(), -1.2, places=5)

    def test_no_clipping_inside_range(self):
        from smalljev.ppo import ppo_policy_loss
        import torch
        logp_new = torch.log(torch.tensor([0.44]))
        logp_old = torch.log(torch.tensor([0.4]))
        adv = torch.tensor([1.0])
        loss = ppo_policy_loss(logp_new, logp_old, adv, clip=0.2)
        self.assertAlmostEqual(loss.item(), -1.1, places=5)

    def test_negative_advantage_clips_downside(self):
        from smalljev.ppo import ppo_policy_loss
        import torch
        # ratio = 0.1, adv = -1: unclipped +0.1... loss = -min(0.1*-1, 0.8*-1) = +0.8
        logp_new = torch.log(torch.tensor([0.04]))
        logp_old = torch.log(torch.tensor([0.4]))
        adv = torch.tensor([-1.0])
        loss = ppo_policy_loss(logp_new, logp_old, adv, clip=0.2)
        self.assertAlmostEqual(loss.item(), 0.8, places=5)

    def test_value_loss_mse(self):
        from smalljev.ppo import value_loss
        import torch
        self.assertAlmostEqual(
            value_loss(torch.tensor([0.5, 0.2]), torch.tensor([1.0, 0.0])).item(),
            0.145, places=5)

    def test_advantage_normalization(self):
        from smalljev.ppo import normalize_advantages
        import torch
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = normalize_advantages(adv)
        self.assertAlmostEqual(out.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(out.std(unbiased=False).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
