import math

def calc_nwgm(token_logprobs, weights=None):
    """
    Calculate the Normalised Weighted Geometric Mean (NWGM) for a sequence of logprobs.
    If weights is None, use uniform weights.
    """
    n = len(token_logprobs)
    if n == 0:
        return float('-inf')
    if weights is None:
        weights = [1.0 / n] * n
    weighted_sum = sum(w * l for w, l in zip(weights, token_logprobs))
    return math.exp(weighted_sum)