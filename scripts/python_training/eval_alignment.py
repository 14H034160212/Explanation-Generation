"""Which test item each stored explanation came from.

`rl_evaluation.evaluate_model` drops an item whose generation came back empty
(`if not explanation: continue`) and, until now, recorded nothing about it. A
short row is therefore a subsequence of the test set with no key saying which
items are missing, and anything that re-pairs by position against the test
file scores explanation j against question j -- the wrong question for every
j after the first drop.

Two halves:

* `test_indices` is written into new eval JSONs by rl_evaluation.py, so the
  question never arises again.
* `recover_indices` reconstructs it for the 38 eval files already on disk.
  ACR is the way back: it is a pure function of (explanation, correct option
  text), and both survive -- the explanation is stored, the option is in the
  test file. So find the increasing index sequence whose recomputed ACR
  matches the stored `acr_scores`. Order is preserved because the evaluation
  loop never reorders, which makes this a subsequence match rather than an
  assignment problem.

The control that makes the recovery trustworthy: a row with no drops must
align to the identity. It does, and when the ACR recomputation was subtly
wrong -- a different correct-option regex -- that control is what failed
first, at index 0, rather than the recovery quietly returning a plausible
wrong answer.
"""
from __future__ import annotations

import re

__all__ = ["correct_option_text", "answer_coverage_rate", "recover_indices"]


def correct_option_text(input_text: str) -> str:
    """Byte for byte what rl_evaluation.extract_correct_option_text returns.

    Imported rather than reimplemented would be better; it lives here because
    importing rl_evaluation pulls in torch and the model stack, and the
    recovery has to run on a machine with no GPU free.
    """
    m = re.search(r"The correct answer is Option ([A-Z])", input_text or "")
    if not m:
        return ""
    letter = m.group(1)
    opt = re.search(
        rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)",
        input_text,
        re.DOTALL,
    )
    return opt.group(1).strip() if opt else ""


def answer_coverage_rate(explanation: str, option_text: str) -> float:
    """Mirrors rl_evaluation.answer_coverage_rate."""
    if not explanation or not option_text:
        return 0.0
    low = explanation.lower()
    terms = re.findall(r"\b\w{4,}\b", option_text.lower())
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in low) / len(terms)


def recover_indices(
    explanations: list[str],
    acr_scores: list[float],
    items: list[dict],
    *,
    tol: float = 1e-9,
) -> list[int] | None:
    """The test index of each explanation, or None if it cannot be pinned down.

    None rather than a best guess on purpose: a caller that cannot align must
    stop, not score something plausible. That is the whole failure being
    fixed.
    """
    if len(explanations) != len(acr_scores) or not explanations:
        return None
    if len(explanations) > len(items):
        return None

    options = [correct_option_text(it.get("input") or "") for it in items]
    out: list[int] = []
    i = 0
    for j, (explanation, want) in enumerate(zip(explanations, acr_scores)):
        # Never look further ahead than the drops still available: without
        # this an early ambiguous match swallows the rest of the row.
        limit = len(items) - (len(explanations) - j) + 1
        hit = None
        while i < limit:
            if abs(answer_coverage_rate(explanation, options[i]) - want) < tol:
                hit = i
                break
            i += 1
        if hit is None:
            return None
        out.append(hit)
        i = hit + 1
    return out
