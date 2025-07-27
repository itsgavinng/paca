"""
Framework Algorithms

This package contains the core algorithms used in the framework.
"""

from .aipw import estimate_causal_effect_aipw
from .cad_analysis import compute_cad_analysis
from .aggregation import aggregate_aspects, generate_abstention_response

__all__ = [
    'estimate_causal_effect_aipw',
    'compute_cad_analysis', 
    'aggregate_aspects',
    'generate_abstention_response'
] 