"""
Grading logic for WhyDidItFail.

grade() is the single entry point. It scores the full episode trajectory:

  diagnosis_score  (0.00 – 0.70)  was the diagnosis correct?
  evidence_score   (0.00 – 0.15)  did the agent inspect the right sources?
  efficiency_score (0.00 – 0.15)  did the agent act without waste?
  fix_bonus        (0.00 – 0.15)  did the agent suggest a valid fix? (bonus, capped at 1.0)

Step-level partial rewards are returned by the environment's step() on every action,
giving the agent a signal over the full trajectory before the episode ends.
"""

# ── keyword maps ──────────────────────────────────────────────────────────────

EXACT_KEYWORDS: dict[str, list[str]] = {
    "exploding_gradients":           ["exploding gradients", "exploding"],
    "learning_rate_too_high":        ["learning rate too high", "lr too high"],
    "overfitting":                   ["overfitting", "overfit"],
    "underfitting":                  ["underfitting", "underfit"],
    "learning_rate_too_low":         ["learning rate too low", "lr too low"],
    "missing_regularization":        ["missing regularization", "no regularization", "lack of regularization"],
    "batch_size_too_small":          ["batch size too small", "small batch size"],
    "optimizer_misconfiguration":    ["optimizer misconfiguration", "optimizer misconfig", "wrong optimizer"],
    "vanishing_gradients":           ["vanishing gradients", "vanishing"],
    "dying_relu":                    ["dying relu", "dead relu"],
    "bad_weight_initialization":     ["bad weight initialization", "poor initialization", "wrong initialization"],
    "lr_scheduler_misconfiguration": ["lr scheduler misconfiguration", "scheduler misconfiguration"],
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "exploding_gradients":           ["nan", "gradient", "overflow", "diverge"],
    "learning_rate_too_high":        ["learning rate", "lr", "oscillat", "unstable"],
    "overfitting":                   ["generalization", "val loss", "memoriz"],
    "underfitting":                  ["plateau", "not learning", "too simple", "high bias"],
    "learning_rate_too_low":         ["slow converge", "converge", "too slow"],
    "missing_regularization":        ["regulariz", "dropout", "weight decay"],
    "batch_size_too_small":          ["batch", "noisy gradient", "gradient noise"],
    "optimizer_misconfiguration":    ["optimizer", "momentum", "sgd"],
    "vanishing_gradients":           ["gradient", "vanish", "sigmoid", "stuck"],
    "dying_relu":                    ["relu", "dead", "zero gradient", "activation"],
    "bad_weight_initialization":     ["initializ", "weight init", "nan"],
    "lr_scheduler_misconfiguration": ["scheduler", "spike", "periodic", "step_lr"],
}


# ── sub-scorers ───────────────────────────────────────────────────────────────

def _diagnosis_score(diagnosis: str, scenario: dict) -> float:
    """
    0.70 — exact keyword match
    0.35 — category / fuzzy match
    0.00 — wrong
    """
    correct = scenario.get("correct_diagnosis", "")
    d = diagnosis.strip().lower()

    score = 0.0

    # exact keyword matches (strong signal)
    for kw in EXACT_KEYWORDS.get(correct, [correct]):
        if kw in d:
            score += 0.4

    # category matches (weaker signal)
    for kw in CATEGORY_KEYWORDS.get(correct, []):
        if kw in d:
            score += 0.1

    # penalize vague answers
    if len(d.split()) < 3:
        score -= 0.1

    return max(0.0, min(0.7, score))


def _evidence_score(inspection_order: list[str], required: set[str]) -> float:
    """
    +0.08 per required source inspected  (max +0.24 for 3 sources)
    −0.06 per required source NOT inspected at submit time
    −0.02 per irrelevant source inspected
    Clamped to [−0.15, +0.25].
    """
    inspected_set = set(inspection_order)
    relevant   = inspected_set & required
    missing    = required - inspected_set
    irrelevant = inspected_set - required

    score = (len(relevant) * 0.08) - (len(missing) * 0.06) - (len(irrelevant) * 0.02)
    return max(-0.15, min(0.25, score))


def _efficiency_score(steps_taken: int, min_steps: int) -> float:
    """
    0.15 at minimum steps, decays −0.025 per extra step, floor 0.0.
    min_steps = number of required sources + 1 (the submit action).
    """
    extra_steps = max(0, steps_taken - min_steps)
    penalty = 0.02 * (extra_steps ** 1.2)
    return max(0.0, 0.15 - penalty)


def _fix_bonus(suggested_fix: str | None, scenario: dict) -> float:
    """
    Bonus score for providing a correct fix. Never penalised for omitting.
    0.15 — all significant keywords from correct_fix are present
    0.08 — at least half the keywords match
    0.00 — no fix or wrong fix
    """
    if not suggested_fix:
        return 0.0

    fix         = suggested_fix.strip().lower()
    correct_fix = scenario.get("correct_fix", "").strip().lower()
    stop        = {"to", "a", "the", "and", "or", "use", "set", "by"}
    keywords    = [w for w in correct_fix.split() if w not in stop and len(w) > 2]

    if not keywords:
        return 0.0

    matched = sum(1 for kw in keywords if kw in fix)

    ratio = matched / len(keywords)

    if ratio == 1.0:
        return 0.15
    elif ratio >= 0.6:
        return 0.10
    elif ratio >= 0.3:
        return 0.05
    return 0.0


# ── main entry point ──────────────────────────────────────────────────────────

def grade(
    diagnosis: str,
    suggested_fix: str | None = None,
    scenario: dict | None = None,
    steps_taken: int = 0,
    inspection_order: list[str] | None = None,
    difficulty: str = "easy",   # kept for API compat — not used in scoring logic
) -> float:
    """
    Single unified grade function. Scores every scenario identically.

    Total score = diagnosis_score + evidence_score + efficiency_score + fix_bonus
                  clamped to [0.0, 1.0].

    Max achievable without fix:  0.70 + 0.15 + 0.15       = 1.00
    Max achievable with fix:     0.70 + 0.15 + 0.15 + 0.15 = 1.00  (capped)
    """
    scenario         = scenario or {}
    inspection_order = inspection_order or []
    required         = set(scenario.get("required_sources", ["logs"]))
    min_steps        = len(required) + 1   # inspect all required sources + submit

    d_score = _diagnosis_score(diagnosis, scenario)
    e_score = _evidence_score(inspection_order, required)
    f_score = _efficiency_score(steps_taken, min_steps)
    b_score = _fix_bonus(suggested_fix, scenario)

    total = d_score + e_score + f_score + b_score

    # bonus if diagnosis and fix are aligned (basic consistency check)
    if suggested_fix and diagnosis:
        if any(word in suggested_fix.lower() for word in diagnosis.lower().split()):
            total += 0.05

    return round(max(0.0, min(1.0, total)), 4)