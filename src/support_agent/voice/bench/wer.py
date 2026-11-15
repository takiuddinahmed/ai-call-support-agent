"""Word error rate."""

import re


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def word_error_rate(reference: str, heard: str) -> float:
    """Returns (substitutions + deletions + insertions) divided by the reference word count.

    The function lowercases both texts and removes punctuation first.
    """
    ref = _words(reference)
    hyp = _words(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)
