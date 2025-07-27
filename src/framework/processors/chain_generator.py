"""
Chain of Thought Generator

Implements the generation of multiple reasoning chains for aspects in Stage 2.
"""

from typing import List
import json
from framework.models import Aspect
from llms import chat
from utils.prompts import GENERATE_COT_PROMPT


def generate_chains_of_thought(question: str, aspect: Aspect, dimension: str, 
                              model: str, k: int = 3) -> List[str]:
    """
    Generate K candidate chains of thought for a given aspect.
    
    Args:
        question: The input question
        aspect: The aspect to reason from
        dimension: The context dimension
        model: LLM model to use
        k: Number of chains to generate
        
    Returns:
        List of chain of thought strings
    """
    chains = []
    
    for i in range(k):
        prompt = GENERATE_COT_PROMPT.format(
            question=question,
            aspect_value=aspect.value,
            dimension=dimension
        )
        
        try:
            response = chat(prompt=prompt, model=model, temperature=0.7)
            
            # Try to parse JSON response
            if response.response.strip().startswith('{'):
                result = json.loads(response.response)
                cot = result.get("CoT", f"Reasoning from {aspect.value} aspect {i+1}")
                chains.append(cot)
            else:
                # Fallback: use the response directly
                cot = response.response.strip() or f"Reasoning from {aspect.value} aspect {i+1}"
                chains.append(cot)
                
        except Exception as e:
            print(f"Error generating CoT {i+1} for {aspect.value}: {e}")
            chains.append(f"Reasoning from {aspect.value} aspect {i+1}")
    
    return chains 