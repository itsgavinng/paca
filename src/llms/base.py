"""
Base interface for LLM models.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from typing import List

class LLMInterface(ABC):
    """Base interface for LLM models."""
    
    @abstractmethod
    def chat(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, Optional[Dict], Optional[float]]:
        """
        Get a response from the LLM.
        
        Args:
            prompt: The input prompt
            model: The model to use
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple of (response, logprobs, nwgm)
            - response: The generated text
            - logprobs: Per-token logprobs or similar info
            - nwgm: Normalised Weighted Geometric Mean of logprobs (float or None)
        """
        pass 

@dataclass
class ChatResponse:
    """Class to encapsulate chat response data."""
    response: str
    conversation_history: List[Dict[str, str]]
    logprobs: Optional[Dict]
    nwgm: Optional[float]

    def __str__(self) -> str:
        return self.response 