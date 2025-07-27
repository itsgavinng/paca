"""
Framework Processors

This package contains processing utilities for chain of thought generation and answer processing.
"""

from .chain_generator import generate_chains_of_thought
from .answer_generator import generate_answer_from_cot

__all__ = [
    'generate_chains_of_thought',
    'generate_answer_from_cot'
] 