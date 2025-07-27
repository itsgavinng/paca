"""
Augmented Inverse Probability Weighting (AIPW) Algorithm

Implementation of the doubly-robust estimator for causal effect estimation
that combines front-door adjustment with bias correction.
"""

from typing import List
from collections import Counter, defaultdict
import numpy as np
from framework.models import Reasoning


def estimate_causal_effect_aipw(reasonings: List[Reasoning]) -> float:
    """
    Compute the Augmented Inverse Probability Weighting (AIPW) causal effect estimate.
    
    This implements the doubly-robust estimator that combines front-door adjustment
    with bias correction.
    
    Args:
        reasonings: List of reasoning objects with answers and probabilities
        
    Returns:
        AIPW causal effect estimate τ̂(x_i)
    """
    if not reasonings:
        return 0.0
    
    # Extract data for AIPW calculation - use actual probabilities as per framework
    cot_indices = [r.cot_position for r in reasonings]
    # Use the probability values from reasonings (log-probabilities or NWGM)
    answer_values = [r.probability for r in reasonings]
    n = len(reasonings)
    
    # Estimate mediator distribution p̂(c_j|x_i)
    cot_counts = Counter(cot_indices)
    p_c = {idx: count / n for idx, count in cot_counts.items()}
    
    # Estimate outcome regression μ̂(c_j|x_i)
    cot_answers = defaultdict(list)
    for reasoning, p_value in zip(reasonings, answer_values):
        cot_answers[reasoning.cot_position].append(p_value)
    
    mu_c = {idx: np.mean(answers) for idx, answers in cot_answers.items()}
    
    # Compute AIPW estimate
    # Front-door term: Σ p̂(c_j|x_i) * μ̂(c_j|x_i)
    front_door = sum(p_c[idx] * mu_c[idx] for idx in p_c.keys())
    
    # Bias correction term: (1/N) * Σ (a_ℓ - μ̂(c_ℓ|x_i)) / p̂(c_ℓ|x_i)
    bias_correction = 0.0
    for reasoning, p_value in zip(reasonings, answer_values):
        cot_idx = reasoning.cot_position
        if p_c[cot_idx] > 0:  # Avoid division by zero
            bias_correction += (p_value - mu_c[cot_idx]) / p_c[cot_idx]
    bias_correction /= n
    
    # Final AIPW estimate
    aipw_estimate = front_door + bias_correction
    
    return float(aipw_estimate) 