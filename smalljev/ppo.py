"""PPO over decision space (cf. prior art: PPO over sequence embeddings).

Policy: categorical over option slots given a sequence embedding h.
Critic: scalar value head predicting expected reward from h.
No token generation anywhere: actions are slot indices, rewards are
correctness / negative-Brier. Used by evals/train_ppo.py for the
SFT vs REINFORCE vs PPO ablation on identical setups.
"""
import torch


def ppo_policy_loss(logp_new, logp_old, advantages, clip=0.2):
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    return -(torch.min(ratio * advantages, clipped * advantages)).mean()


def value_loss(v_pred, returns):
    return torch.nn.functional.mse_loss(v_pred, returns)


def normalize_advantages(advantages, eps=1e-8):
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def explained_variance(v_pred, returns):
    var_y = returns.var(unbiased=False)
    if var_y.item() == 0:
        return float("nan")
    return float(1.0 - ((returns - v_pred).var(unbiased=False) / var_y).item())
