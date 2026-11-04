"""decide(): STATE + decision spec -> tiny typed JSON. Zero generated tokens."""
from .model import get_backend
from .spec import ChoiceQuestion, NoulQuestion, build_spec


def _as_items(questions):
    if isinstance(questions, dict):
        return [{"name": name, **spec} for name, spec in questions.items()]
    return list(questions)


def decide(state, questions, backend="stub", temperature=1.0):
    spec = build_spec(_as_items(questions))
    be = get_backend(backend)
    prob_table = be.score_all(state, spec, temperature=temperature)
    out = {}
    for name, q in spec.items():
        if isinstance(q, NoulQuestion):
            out[name] = {"probability": float(prob_table[name])}
            continue
        probs = [float(p) for p in prob_table[name]]
        if isinstance(q, ChoiceQuestion):
            out[name] = {"values": list(q.choices), "probabilities": probs}
        else:
            value = float(sum(p * v for p, v in zip(probs, q.values)))
            out[name] = {"value": value, "probabilities": probs,
                         "levels": list(q.levels)}
    out["_meta"] = {"backend": getattr(be, "name", "custom"),
                    "generated_tokens": int(getattr(be, "generated_tokens", 0)),
                    "n_questions": len(spec)}
    return out


def trajectory(turns, questions, backend="stub", temperature=1.0):
    """Score each conversation PREFIX turns[:i] as its own state (paper-1 sequential
    formulation: states, probability-estimate actions, accuracy rewards).

    Returns one decide() output per prefix. Prefixes share string content, so a
    serving backend can reuse KVs incrementally instead of re-encoding.
    Machinery demo-grade: use Banking77/agnews-style labeled outcomes per prefix
    for real trajectory eval (e.g. saas-sales-conversations, Apache-2.0, deferred).
    """
    states = ["\n".join(turns[:i]) for i in range(1, len(turns) + 1)]
    return [decide(s, questions, backend=backend, temperature=temperature)
            for s in states]
