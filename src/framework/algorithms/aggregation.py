"""
Answer Aggregation and Abstention Response Generation

Implements the logic for aggregating answers from multiple aspects and generating
appropriate abstention responses when needed.
"""

from typing import List
import json
from framework.models import ProcessedAspect
from llms import chat
from utils.prompts import (
    AGGREGATE_ANSWERS_PROMPT,
    ABSTAIN_TYPE1_PROMPT,
    ABSTAIN_TYPE2_PROMPT
)


def aggregate_aspects(question: str, processed_aspects: List[ProcessedAspect], 
                     model: str) -> str:
    """
    Aggregate answers from multiple aspects using significance weighting.
    
    Args:
        question: The input question
        processed_aspects: List of processed aspects
        model: LLM model to use for aggregation
        
    Returns:
        Aggregated final answer
    """
    # Create summary of aspects and their significant answers
    aspects_summary = []
    
    for proc_aspect in processed_aspects:
        significance = proc_aspect.aspect.weight * proc_aspect.causal_effect
        
        if proc_aspect.reasonings:
            best_reasoning = max(proc_aspect.reasonings, key=lambda r: r.probability)
            representative_answer = best_reasoning.answer
            all_answers = [r.answer for r in proc_aspect.reasonings]
        else:
            representative_answer = "No answer generated"
            all_answers = []
        
        summary = f"""
Aspect: {proc_aspect.aspect.value}
Weight: {proc_aspect.aspect.weight:.3f}
Causal Effect: {proc_aspect.causal_effect:.3f}
Significance (α): {significance:.3f}
Representative Answer: {representative_answer}
All Answers: {all_answers}
"""
        aspects_summary.append(summary)
    
    prompt = AGGREGATE_ANSWERS_PROMPT.format(
        question=question,
        aspects_summary="\n".join(aspects_summary)
    )
    
    try:
        response = chat(prompt=prompt, model=model, temperature=0.3)
        
        # Try to parse JSON response
        if response.response.strip().startswith('{'):
            result = json.loads(response.response)
            return result.get("final_answer", response.response)
        else:
            return response.response.strip()
            
    except Exception as e:
        print(f"Error in aggregation: {e}")
        return "Error: Could not synthesize answers from different aspects"


def generate_abstention_response(question: str, abstention_type: str, 
                                processed_aspects: List[ProcessedAspect],
                                model: str) -> str:
    """
    Generate an appropriate abstention response.
    
    Args:
        question: The input question
        abstention_type: "TYPE1" (contradiction) or "TYPE2" (insufficiency)
        processed_aspects: List of processed aspects
        model: LLM model to use
        
    Returns:
        Abstention explanation
    """
    if abstention_type == "TYPE1":
        # Knowledge contradiction
        conflict_details = "High angular deviation detected between aspects: "
        conflict_details += ", ".join([p.aspect.value for p in processed_aspects])
        
        prompt = ABSTAIN_TYPE1_PROMPT.format(
            question=question,
            conflict_details=conflict_details
        )
    else:  # TYPE2
        # Knowledge insufficiency
        insufficiency_details = "Analysis converges toward uncertainty across aspects: "
        insufficiency_details += ", ".join([p.aspect.value for p in processed_aspects])
        
        prompt = ABSTAIN_TYPE2_PROMPT.format(
            question=question,
            insufficiency_details=insufficiency_details
        )
    
    try:
        response = chat(prompt=prompt, model=model)
        
        # Try to parse JSON response
        if response.response.strip().startswith('{'):
            result = json.loads(response.response)
            return result.get("final_answer", response.response)
        else:
            return response.response.strip()
            
    except Exception as e:
        print(f"Error generating abstention response: {e}")
        if abstention_type == "TYPE1":
            return "I cannot provide a definitive answer due to conflicting information from different aspects."
        else:
            return "I don't have sufficient information to answer this question confidently." 