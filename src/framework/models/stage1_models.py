"""
Stage 1 Data Models

Contains all data models used in the Dual-Agent Aspect Discovery process.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, RootModel


class Dimension(BaseModel):
    """Represents a discovered dimension/context variable."""
    name: str
    description: str
    justification: str
    score: float = 0.0
    
    
class Aspect(BaseModel):
    """Represents an aspect/stratum within a dimension."""
    value: str
    description: str
    justification: str
    weight: float = 0.0


class AspectWeight(BaseModel):
    """Individual aspect weight assignment."""
    value: str
    weight: float
    justification: str


class DiscoveryResult(BaseModel):
    """Final result of the aspect discovery process."""
    best_dimension: Dimension
    aspects: List[Aspect]
    final_weights: Dict[str, float]


# Root models for LangChain parsing
class DimensionList(RootModel):
    """Root model for parsing list of dimensions."""
    root: List[Dimension]


class AspectList(RootModel):
    """Root model for parsing list of aspects."""
    root: List[Aspect]


class AspectWeightList(RootModel):
    """Root model for parsing list of aspect weights."""
    root: List[AspectWeight] 