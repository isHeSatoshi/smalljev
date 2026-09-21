"""v1 learned decision heads: option slots bound per-input, no token priors.

A SlotChoiceHead maps a readout hidden state h to a distribution over option
SLOTS (slot i = i-th option string in THIS input). Unlike the v0 LM-head
verbalizer, slots carry no pretrained letter preference: at zero init the head
is exactly uniform for any option set (tested). Invalid slots are sliced away,
never merely masked — they cannot receive probability.
"""
import torch


def find_spans(full_ids, option_ids_list):
    """Locate each option's token span sequentially in the concatenated stream.

    Returns [(start, end)] with end exclusive. A missing option yields an empty
    span (start == end) as an explicit fallback marker for the caller to handle.
    """
    spans, pos = [], 0
    full = list(full_ids)
    for oids in option_ids_list:
        oids = list(oids)
        found = None
        for s in range(pos, len(full) - len(oids) + 1):
            if full[s:s + len(oids)] == oids:
                found = s
                break
        if found is None:
            spans.append((pos, pos))
        else:
            spans.append((found, found + len(oids)))
            pos = found + len(oids)
    return spans


class OptionScorerHead(torch.nn.Module):
    """Shared per-option scorer: same weights score every option's representation.

    Slot-binding comes from the INPUTS (which span), never the weights — so the
    head is permutation-equivariant by construction and has no positional bias.
    Zero-init gives exactly uniform probabilities for any option set.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(self.scorer.weight)
        torch.nn.init.zeros_(self.scorer.bias)

    def logits(self, H):
        return self.scorer(H).squeeze(-1)

    def probs(self, H):
        return torch.softmax(self.logits(H), dim=-1)

    def save(self, path):
        torch.save({"hidden_size": self.scorer.in_features,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=True)
        head = cls(blob["hidden_size"])
        head.load_state_dict(blob["state_dict"])
        return head


class SlotChoiceHead(torch.nn.Module):
    def __init__(self, hidden_size, max_slots=64):
        super().__init__()
        self.slot = torch.nn.Linear(hidden_size, max_slots)
        torch.nn.init.zeros_(self.slot.weight)
        torch.nn.init.zeros_(self.slot.bias)
        self.max_slots = max_slots

    def logits(self, h, k):
        if k > self.max_slots:
            raise ValueError(f"{k} options exceeds max_slots={self.max_slots}; "
                             "use 2-stage retrieval.")
        return self.slot(h)[..., :k]

    def probs(self, h, k):
        return torch.softmax(self.logits(h, k), dim=-1)

    def save(self, path):
        torch.save({"hidden_size": self.slot.in_features,
                    "max_slots": self.max_slots,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=True)
        head = cls(blob["hidden_size"], blob["max_slots"])
        head.load_state_dict(blob["state_dict"])
        return head


class BinaryNoulHead(torch.nn.Module):
    """P(statement is true). Zero-init bias → exactly 0.5 for any input."""

    def __init__(self, hidden_size):
        super().__init__()
        self.bit = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(self.bit.weight)
        torch.nn.init.zeros_(self.bit.bias)

    def prob(self, h):
        return torch.sigmoid(self.bit(h)).squeeze(-1)


class OrdinalScoreHead(torch.nn.Module):
    """Distribution over the first L of max_levels ordered levels + expectation.

    Level semantics ride in the input (like option slots); slot i = i-th level.
    """

    def __init__(self, hidden_size, max_levels=8):
        super().__init__()
        self.level = torch.nn.Linear(hidden_size, max_levels)
        torch.nn.init.zeros_(self.level.weight)
        torch.nn.init.zeros_(self.level.bias)
        self.max_levels = max_levels

    def logits(self, h, n_levels):
        if n_levels > self.max_levels:
            raise ValueError(f"{n_levels} levels exceeds max_levels={self.max_levels}.")
        return self.level(h)[..., :n_levels]

    def probs(self, h, n_levels):
        return torch.softmax(self.logits(h, n_levels), dim=-1)

    def expected(self, h, n_levels, values):
        p = self.probs(h, n_levels)
        v = torch.tensor(list(values), dtype=p.dtype, device=p.device)
        return (p * v).sum(-1)


class HeadsBundle(torch.nn.Module):
    """All three Jev-shaped primitives: Choice slots + Noul bit + Score levels."""

    def __init__(self, hidden_size, max_slots=64, max_levels=8):
        super().__init__()
        self.choice = SlotChoiceHead(hidden_size, max_slots)
        self.noul = BinaryNoulHead(hidden_size)
        self.score = OrdinalScoreHead(hidden_size, max_levels)

    def save(self, path):
        torch.save({"hidden_size": self.choice.slot.in_features,
                    "max_slots": self.choice.max_slots,
                    "max_levels": self.score.max_levels,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=True)
        bundle = cls(blob["hidden_size"], blob["max_slots"], blob["max_levels"])
        bundle.load_state_dict(blob["state_dict"])
        return bundle
