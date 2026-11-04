"""Typed decision specifications: Choice / Score. Zero weights, zero inference."""
from dataclasses import dataclass, field

MAX_CHOICES = 255  # same cap Jev documents; beyond this use 2-stage retrieval


@dataclass(frozen=True)
class ChoiceQuestion:
    name: str
    question: str
    choices: tuple

    def __init__(self, name, question, choices):
        choices = tuple(choices)
        if len(choices) < 2:
            raise ValueError(f"ChoiceQuestion '{name}' needs >= 2 options.")
        if len(choices) > MAX_CHOICES:
            raise ValueError(f"ChoiceQuestion '{name}' exceeds {MAX_CHOICES} options; "
                             "use 2-stage retrieval instead.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "choices", choices)


@dataclass(frozen=True)
class NoulQuestion:
    """Yes/no statement scored as P(true). Jev's Noul primitive."""
    name: str
    question: str

    def __init__(self, name, question):
        if not question:
            raise ValueError(f"NoulQuestion '{name}' needs a statement.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "question", question)


@dataclass(frozen=True)
class ScoreQuestion:
    name: str
    question: str
    levels: tuple
    values: tuple

    def __init__(self, name, question, levels=("very low", "low", "medium",
                                              "high", "very high"),
                 values=(0.0, 0.25, 0.5, 0.75, 1.0)):
        levels, values = tuple(levels), tuple(values)
        if len(levels) != len(values) or len(levels) < 2:
            raise ValueError(f"ScoreQuestion '{name}' needs >= 2 levels with values.")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"ScoreQuestion '{name}' values must be strictly increasing.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "values", values)


def build_spec(items):
    """Validate a list of plain dicts into an ordered {name: question} mapping."""
    spec = {}
    for item in items:
        kind = item.get("type")
        name = item.get("name")
        if name in spec:
            raise ValueError(f"Duplicate question name: '{name}'.")
        if kind == "choice":
            spec[name] = ChoiceQuestion(name, item.get("question", ""),
                                        item.get("choices", ()))
        elif kind == "noul":
            spec[name] = NoulQuestion(name, item.get("question", ""))
        elif kind == "score":
            levels = item.get("levels")
            values = item.get("values")
            low = item.get("low", 0.0)
            high = item.get("high", 1.0)
            if levels is None:  # continuous-style {"low":0,"high":1} -> 5 ordinal bins
                n = 5
                levels = ("very low", "low", "medium", "high", "very high")
                values = tuple(low + (high - low) * i / (n - 1) for i in range(n))
            spec[name] = ScoreQuestion(name, item.get("question", ""),
                                       levels, values)
        else:
            raise ValueError(f"Unknown question type: {kind!r}.")
    return spec
