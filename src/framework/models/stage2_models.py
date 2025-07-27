"""
Stage 2 Data Models

Contains all data models used in the Aspect Resolution process.
"""

from typing import List, Optional
from dataclasses import dataclass
from .stage1_models import Aspect


@dataclass
class Reasoning:
    """A reasoning chain with its answer and probability."""
    cot_position: int
    answer: str
    probability: float
    
    def to_dict(self):
        return {
            "cot_position": self.cot_position,
            "answer": self.answer,
            "probability": self.probability
        }


@dataclass
class ProcessedAspect:
    """An aspect with processed reasoning chains and causal effect."""
    aspect: Aspect
    chains_of_thought: List[str]
    reasonings: List[Reasoning]
    causal_effect: float  # τ̂(x_i) from AIPW
    
    def to_dict(self):
        return {
            "aspect": {
                "value": self.aspect.value,
                "description": self.aspect.description,
                "weight": self.aspect.weight
            },
            "chains_of_thought": self.chains_of_thought,
            "reasonings": [r.to_dict() for r in self.reasonings],
            "causal_effect": self.causal_effect
        }


@dataclass
class Stage2Result:
    """Result of Stage 2 processing."""
    question: str
    dimension: str
    processed_aspects: List[ProcessedAspect]
    final_answer: str
    abstention_type: Optional[str]  # None, "TYPE1", "TYPE2"
    cad_score: float
    centroid_null_similarity: float
    
    def to_dict(self):
        return {
            "question": self.question,
            "dimension": self.dimension,
            "processed_aspects": [p.to_dict() for p in self.processed_aspects],
            "final_answer": self.final_answer,
            "abstention_type": self.abstention_type,
            "cad_score": self.cad_score,
            "centroid_null_similarity": self.centroid_null_similarity
        } 