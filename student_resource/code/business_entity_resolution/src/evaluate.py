"""Macro F0.5 (beta=0.5): per-S1 F, averaged. Singletons included."""
from __future__ import annotations


def f05_for_entity(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(pred: dict[str, set[str]], gt: dict[str, set[str]]) -> float:
    tot, n = 0.0, 0
    for sid, true in gt.items():
        tot += f05_for_entity(pred.get(sid, set()), true)
        n += 1
    return tot / n if n else 0.0


def sweep_threshold(
    scores: dict[str, list[tuple[str, float]]],
    gt: dict[str, set[str]],
    thresholds: list[float] | None = None,
) -> tuple[float, float]:
    """Pick threshold maximising macro F0.5. Returns (best_thr, best_f05)."""
    if thresholds is None:
        thresholds = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30..0.90
    best_t, best_f = thresholds[0], -1.0
    for t in thresholds:
        pred = {sid: {c for c, s in pairs if s >= t} for sid, pairs in scores.items()}
        f = macro_f05(pred, {k: v for k, v in gt.items() if k in scores})
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f
