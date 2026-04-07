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


def _diagnosis_score(diagnosis: str, scenario: dict) -> float:
    """
    0.70 — exact keyword match
    0.35 — category / fuzzy match
    0.00 — wrong
    """
    correct = scenario.get("correct_diagnosis", "")
    d = diagnosis.strip().lower()

    score = 0.0
    exact_matched = False

    # exact keyword matches (strong signal)
    for kw in EXACT_KEYWORDS.get(correct, [correct]):
        if kw in d:
            score += 0.4
            exact_matched = True

    # category matches (weaker signal)
    for kw in CATEGORY_KEYWORDS.get(correct, []):
        if kw in d:
            score += 0.1

    # penalize vague answers only when no exact match found
    if not exact_matched and len(d.split()) < 3:
        score -= 0.1

    return max(0.0, min(0.7, score))


def _evidence_diagnosis_penalty(
    diagnosis: str,
    scenario: dict,
    inspection_order: list[str],
) -> float:
    """
    Penalises reasoning failure: the agent had evidence but drew the wrong conclusion.

    All required sources inspected, wrong diagnosis:  −0.10  (clear reasoning failure)
    Some required sources inspected, wrong diagnosis: −0.05  (partial reasoning failure)
    No required sources inspected, wrong diagnosis:    0.00  (evidence_score handles this)
    Correct diagnosis:                                 0.00  (no penalty)
    """
    correct = scenario.get("correct_diagnosis", "")
    d = diagnosis.strip().lower()

    is_correct = any(kw in d for kw in EXACT_KEYWORDS.get(correct, [correct]))
    if is_correct:
        return 0.0

    required         = set(scenario.get("required_sources", ["logs"]))
    inspected_req    = set(inspection_order) & required

    if inspected_req == required:
        return -0.10   # had everything, still wrong
    if inspected_req:
        return -0.05   # partial evidence, still wrong
    return 0.0         # blind submission — evidence_score already penalises


def _evidence_score(inspection_order: list[str], required: set[str]) -> float:
    """
    +0.08 per required source inspected  (max +0.24 for 3 sources)
    −0.10 for each required source NOT inspected at submit time
    −0.02 per irrelevant source inspected
    Clamped to [−0.15, +0.25].
    """
    inspected_set = set(inspection_order)
    relevant   = inspected_set & required
    missing    = required - inspected_set
    irrelevant = inspected_set - required

    score = (len(relevant) * 0.08) - (len(missing) * 0.10) - (len(irrelevant) * 0.02)
    return max(-0.15, min(0.25, score))


def _efficiency_score(steps_taken: int, min_steps: int) -> float:
    """
    0.15 at minimum steps.
    Decays for extra steps (wasted) or missing steps (early submission).
    min_steps = number of required sources + 1 (the submit action).
    """
    if steps_taken < min_steps:
        missing_steps = min_steps - steps_taken
        return max(0.0, 0.15 - 0.05 * missing_steps)
    extra_steps = steps_taken - min_steps
    penalty = 0.02 * (extra_steps ** 1.2)
    return max(0.0, 0.15 - penalty)


def _fix_score(suggested_fix: str | None, scenario: dict) -> float:
    """
    Uniform fix score applied to every scenario.

      −0.05 — no fix provided          (always expected)
       0.00 — fix provided but wrong
      +0.05 — ≥30% keyword match
      +0.10 — ≥60% keyword match
      +0.15 — all keywords match
    """
    if not suggested_fix:
        return -0.05   # fix is always expected; omitting it costs points

    fix         = suggested_fix.strip().lower()
    correct_fix = scenario.get("correct_fix", "").strip().lower()
    stop        = {"to", "a", "the", "and", "or", "use", "set", "by"}
    keywords    = [w for w in correct_fix.split() if w not in stop and len(w) > 2]

    if not keywords:
        return 0.0

    ratio = sum(1 for kw in keywords if kw in fix) / len(keywords)

    if ratio == 1.0:
        return 0.15
    elif ratio >= 0.6:
        return 0.10
    elif ratio >= 0.3:
        return 0.05
    return 0.0


def _ordering_bonus(inspection_order: list[str], required_sources: list[str]) -> float:
    """
    +0.05 if the agent inspected required sources in the canonical order.

    Canonical order is defined by required_sources in the scenario
    (always a prefix of logs → config → gradients).

    Only the order of required sources matters — irrelevant sources inspected
    in between are ignored when checking sequence.
    """
    required_set = set(required_sources)
    # Subsequence of inspection_order containing only required sources
    inspected_required = [s for s in inspection_order if s in required_set]

    # Check against canonical order, limited to what was actually inspected
    canonical = [s for s in required_sources if s in inspected_required]

    return 0.05 if inspected_required == canonical else 0.0


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
    required_sources = scenario.get("required_sources", ["logs"])   # ordered list
    required         = set(required_sources)                        # set for membership checks
    min_steps        = len(required) + 1          # inspect all required sources + submit
    max_steps        = len(required) * 3 + 2      # hard ceiling; exceeding it = total failure

    if steps_taken > max_steps:
        return 0.0

    d_score  = _diagnosis_score(diagnosis, scenario)
    ed_penalty = _evidence_diagnosis_penalty(diagnosis, scenario, inspection_order)
    e_score  = _evidence_score(inspection_order, required)
    f_score  = _efficiency_score(steps_taken, min_steps)
    b_score  = _fix_score(suggested_fix, scenario)
    o_bonus  = _ordering_bonus(inspection_order, required_sources)

    total = d_score + ed_penalty + e_score + f_score + b_score + o_bonus

    return round(max(0.0, min(1.0, total)), 4)