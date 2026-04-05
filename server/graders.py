_STOP_WORDS = {"to", "a", "the", "and", "or", "is", "are", "was", "an", "in", "of"}

def _normalize(s: str) -> str:
    return s.replace("_", " ").replace("-", " ")

def _keywords_match(submitted: str, expected: str) -> bool:
    """Return True if all significant keywords from expected appear in submitted."""
    submitted_norm = _normalize(submitted)
    keywords = [w for w in _normalize(expected).split() if w not in _STOP_WORDS and len(w) > 1]
    return all(kw in submitted_norm for kw in keywords)

def grade_easy(diagnosis: str, scenario: dict) -> float:
    """Easy: keyword match against correct_diagnosis."""
    return 1.0 if _keywords_match(diagnosis.strip().lower(), scenario["correct_diagnosis"].strip().lower()) else 0.0

def grade_medium(diagnosis: str, scenario: dict) -> float:
    """Medium: Did the agent identify the correct failure category?"""
    correct = scenario["correct_diagnosis"]   # e.g. "exploding_gradients"
    aliases = {
        "exploding_gradients": ["explod", "nan loss", "gradient", "lr too high"],
        "overfitting":         ["overfit", "val loss", "generaliz"],
        "vanishing_gradients": ["vanish", "gradient", "dead", "stuck"],
        "underfitting":        ["underfit", "too simple", "high bias", "plateau"],
        "lr_too_low":          ["lr too low", "learning rate", "slow converge"],
    }
    matches = aliases.get(correct, [correct])
    for m in matches:
        if m in diagnosis.lower():
            return 1.0
    # partial credit for being in the right ballpark
    if "gradient" in diagnosis.lower() and "gradient" in correct:
        return 0.5
    return 0.0

def grade_hard(diagnosis: str, fix: str, scenario: dict) -> float:
    """Hard: Correct diagnosis + correct fix with evidence."""
    diagnosis_score = grade_medium(diagnosis, scenario)
    correct_fix = scenario["correct_fix"]
    fix_score = 0.0
    if fix and correct_fix.split()[0] in fix.lower():   # checks verb
        fix_score += 0.5
    if fix and any(w in fix.lower() for w in correct_fix.split()):
        fix_score += 0.5
    # Bonus: agent must have inspected both logs AND config
    # TODO : 
    investigation_bonus = 0.0  # passed from environment
    return min(1.0, (diagnosis_score * 0.5) + (min(fix_score, 1.0) * 0.5))