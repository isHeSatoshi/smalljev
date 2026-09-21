"""v1 shared-prefix sampler primitives.

Packs [STATE][B1]...[Bk] into ONE sequence with a block-isolation mask:
branch Bi attends STATE + its own past, never another branch. STATE is encoded
once; branches add only suffix cost. No weights needed for mask construction.
"""
import numpy as np

BLOCKED = -3.0e4  # additive -inf that survives bf16/fp16/fp32


def build_block_mask(state_len, branch_lens):
    """Boolean (T, T) allow-mask. Query p attends key k iff k<=p (causal) and
    not (p in Bi and k in Bj with i != j)."""
    branch_lens = list(branch_lens)
    total = state_len + sum(branch_lens)
    bid = np.full(total, -1, dtype=np.int64)
    s = state_len
    for i, length in enumerate(branch_lens):
        bid[s:s + length] = i
        s += length
    q = np.arange(total)[:, None]
    k = np.arange(total)[None, :]
    causal = k <= q
    cross = (bid[:, None] >= 0) & (bid[None, :] >= 0) & (bid[:, None] != bid[None, :])
    return causal & ~cross


def readouts(state_len, branch_lens):
    """Index of the last token of each branch (= decision readout position)."""
    pos, out = state_len, []
    for length in branch_lens:
        pos += length
        out.append(pos - 1)
    return out


def to_additive_4d(allow):
    """(T, T) bool -> (1, 1, T, T) float additive mask for HF eager/sdpa."""
    import torch
    add = torch.full((allow.shape[0], allow.shape[1]), BLOCKED, dtype=torch.float32)
    add[torch.from_numpy(allow)] = 0.0
    return add.unsqueeze(0).unsqueeze(0)
