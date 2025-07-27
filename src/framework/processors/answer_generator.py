"""
Answer Generator

Implements the generation of answers from chains of thought for aspects in Stage 2.
"""

import json
from framework.models import Aspect
from llms import chat
from utils.prompts import ANSWER_FROM_COT_PROMPT


def generate_answer_from_cot(question: str, aspect: Aspect, cot: str, 
                            model: str) -> str:
    """
    Generate an answer based on a specific chain of thought.
    
    Args:
        question: The input question
        aspect: The aspect context
        cot: Chain of thought to reason from
        model: LLM model to use
        
    Returns:
        Generated answer string
    """
    prompt = ANSWER_FROM_COT_PROMPT.format(
        question=question,
        aspect_value=aspect.value,
        CoT=cot
    )
    
    try:
        response = chat(prompt=prompt, model=model, temperature=0.3)
        
        # Try to parse JSON response
        if response.response.strip().startswith('{'):
            result = json.loads(response.response)
            return result.get("answer", "No answer generated")
        else:
            # Return the raw response if not JSON
            return response.response.strip() or "No answer generated"
            
    except Exception as e:
        print(f"Error generating answer from CoT: {e}")
        return "Error generating answer" 