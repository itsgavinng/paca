"""
Stage 2: Perspective Resolution

This module implements the second stage of the framework that answers:
"How much should I trust each perspective?"

Key components:
1. Causal Effect Estimation using Augmented Inverse Probability Weighting (AIPW)
2. Abstention Policy using Centroid Angular Deviation (CAD) analysis
3. Three-way decision gate: Type-1 Abstention, Type-2 Abstention, or Aggregation
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import random
from pydantic import BaseModel, Field

# Handle sentence transformers import with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import framework components
from framework.stage_1 import DiscoveryResult, Perspective
from llms import chat
from llms.base import ChatResponse
from utils.prompts import (
    GENERATE_COT_PROMPT,
    ANSWER_FROM_COT_PROMPT,
    AGGREGATE_ANSWERS_PROMPT,
    ABSTAIN_TYPE1_PROMPT,
    ABSTAIN_TYPE2_PROMPT
)


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
class ProcessedPerspective:
    """A perspective with processed reasoning chains and causal effect."""
    perspective: Perspective
    chains_of_thought: List[str]
    reasonings: List[Reasoning]
    causal_effect: float  # τ̂(x_i) from AIPW
    
    def to_dict(self):
        return {
            "perspective": {
                "value": self.perspective.value,
                "description": self.perspective.description,
                "weight": self.perspective.weight
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
    processed_perspectives: List[ProcessedPerspective]
    final_answer: str
    abstention_type: Optional[str]  # None, "TYPE1", "TYPE2"
    cad_score: float
    centroid_null_similarity: float
    
    def to_dict(self):
        return {
            "question": self.question,
            "dimension": self.dimension,
            "processed_perspectives": [p.to_dict() for p in self.processed_perspectives],
            "final_answer": self.final_answer,
            "abstention_type": self.abstention_type,
            "cad_score": self.cad_score,
            "centroid_null_similarity": self.centroid_null_similarity
        }





def generate_chains_of_thought(question: str, perspective: Perspective, dimension: str, 
                              model: str, k: int = 3) -> List[str]:
    """
    Generate K candidate chains of thought for a given perspective.
    
    Args:
        question: The input question
        perspective: The perspective to reason from
        dimension: The context dimension
        model: LLM model to use
        k: Number of chains to generate
        
    Returns:
        List of chain of thought strings
    """
    prompt = GENERATE_COT_PROMPT.format(
        question=question,
        perspective_value=perspective.value,
        dimension=dimension,
        k=k
    )
    
    try:
        response = chat(prompt=prompt, model=model, temperature=0.7)
        
        # Try to parse JSON response
        if response.response.strip().startswith('{'):
            result = json.loads(response.response)
            return result.get("chains_of_thought", [])[:k]
        else:
            # Fallback: split by newlines and take meaningful lines
            lines = [line.strip() for line in response.response.split('\n') if line.strip()]
            return lines[:k] if lines else [f"Reasoning from {perspective.value} perspective"]
            
    except Exception as e:
        print(f"Error generating CoT for {perspective.value}: {e}")
        return [f"Reasoning from {perspective.value} perspective"]


def generate_answer_from_cot(question: str, perspective: Perspective, cot: str, 
                            model: str) -> str:
    """
    Generate an answer based on a specific chain of thought.
    
    Args:
        question: The input question
        perspective: The perspective context
        cot: Chain of thought to reason from
        model: LLM model to use
        
    Returns:
        Generated answer string
    """
    prompt = ANSWER_FROM_COT_PROMPT.format(
        question=question,
        perspective_value=perspective.value,
        cot=cot
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


def compute_cad_analysis(processed_perspectives: List[ProcessedPerspective]) -> Tuple[float, float, List[float]]:
    """
    Compute Centroid Angular Deviation (CAD) analysis for abstention policy.
    
    Args:
        processed_perspectives: List of processed perspectives with causal effects
        
    Returns:
        Tuple of (CAD score, centroid-null similarity, angular deviations)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Fallback: use simple heuristics
        print("Warning: Using simplified CAD analysis without sentence transformers")
        return 0.3, 0.5, [0.2] * len(processed_perspectives)
    
    try:
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get representative answers for each perspective
        representative_answers = []
        weights = []
        causal_effects = []
        
        for proc_persp in processed_perspectives:
            if proc_persp.reasonings:
                # Get the answer with highest probability as representative
                best_reasoning = max(proc_persp.reasonings, key=lambda r: r.probability)
                representative_answers.append(best_reasoning.answer)
            else:
                representative_answers.append("No answer generated")
            
            weights.append(proc_persp.perspective.weight)
            causal_effects.append(proc_persp.causal_effect)
        
        # Get embeddings and normalize
        embeddings = embedding_model.encode(representative_answers)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute significance α_i = w_i * τ̂(x_i)
        significance = np.array(weights) * np.array(causal_effects)
        
        # Compute causally weighted centroid
        if np.sum(significance) > 1e-8:
            centroid_raw = np.sum(significance.reshape(-1, 1) * embeddings, axis=0)
            centroid = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)
        else:
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        # Compute angular deviations
        angular_deviations = []
        for embedding in embeddings:
            dot_product = np.clip(np.dot(embedding, centroid), -1.0, 1.0)
            theta = np.arccos(dot_product)
            angular_deviations.append(theta)
        
        angular_deviations = np.array(angular_deviations)
        
        # Compute CAD score
        if np.sum(significance) > 1e-8:
            cad_score = np.sum(significance * angular_deviations) / np.sum(significance)
        else:
            cad_score = np.mean(angular_deviations)
        
        # Compute centroid-null similarity - using standardized uncertainty expressions
        null_phrases = [
            "cannot be determined",
            "insufficient evidence", 
            "unknowable from this perspective",
            "cannot be determined from this perspective",
            "insufficient evidence from this perspective",
            "unknowable",
            "cannot be established",
            "insufficient information",
            "no conclusive evidence",
            "cannot be resolved",
            "indeterminate",
            "uncertain",
            "unclear",
            "no data",
            "no information",
            "no evidence",
            "no answer",
            "no conclusion",
            "no determination",
            "no resolution",
            "no answer"
        ]        
        null_embeddings = embedding_model.encode(null_phrases)
        null_consensus = np.mean(null_embeddings, axis=0)
        null_consensus = null_consensus / (np.linalg.norm(null_consensus) + 1e-8)
        
        centroid_null_similarity = np.clip(np.dot(centroid, null_consensus), -1.0, 1.0)
        
        return float(cad_score), float(centroid_null_similarity), angular_deviations.tolist()
        
    except Exception as e:
        print(f"Error in CAD analysis: {e}")
        # Return default values
        return 0.3, 0.5, [0.2] * len(processed_perspectives)


def aggregate_perspectives(question: str, processed_perspectives: List[ProcessedPerspective], 
                          model: str) -> str:
    """
    Aggregate answers from multiple perspectives using significance weighting.
    
    Args:
        question: The input question
        processed_perspectives: List of processed perspectives
        model: LLM model to use for aggregation
        
    Returns:
        Aggregated final answer
    """
    # Create summary of perspectives and their significant answers
    perspectives_summary = []
    
    for proc_persp in processed_perspectives:
        significance = proc_persp.perspective.weight * proc_persp.causal_effect
        
        if proc_persp.reasonings:
            best_reasoning = max(proc_persp.reasonings, key=lambda r: r.probability)
            representative_answer = best_reasoning.answer
            all_answers = [r.answer for r in proc_persp.reasonings]
        else:
            representative_answer = "No answer generated"
            all_answers = []
        
        summary = f"""
Perspective: {proc_persp.perspective.value}
Weight: {proc_persp.perspective.weight:.3f}
Causal Effect: {proc_persp.causal_effect:.3f}
Significance (α): {significance:.3f}
Representative Answer: {representative_answer}
All Answers: {all_answers}
"""
        perspectives_summary.append(summary)
    
    prompt = AGGREGATE_ANSWERS_PROMPT.format(
        question=question,
        perspectives_summary="\n".join(perspectives_summary)
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
        return "Error: Could not synthesize answers from different perspectives"


def generate_abstention_response(question: str, abstention_type: str, 
                                processed_perspectives: List[ProcessedPerspective],
                                model: str) -> str:
    """
    Generate an appropriate abstention response.
    
    Args:
        question: The input question
        abstention_type: "TYPE1" (contradiction) or "TYPE2" (insufficiency)
        processed_perspectives: List of processed perspectives
        model: LLM model to use
        
    Returns:
        Abstention explanation
    """
    if abstention_type == "TYPE1":
        # Knowledge contradiction
        contradiction_details = "High angular deviation detected between perspectives: "
        contradiction_details += ", ".join([p.perspective.value for p in processed_perspectives])
        
        prompt = ABSTAIN_TYPE1_PROMPT.format(
            question=question,
            contradiction_details=contradiction_details
        )
    else:  # TYPE2
        # Knowledge insufficiency
        insufficiency_details = "Analysis converges toward uncertainty across perspectives: "
        insufficiency_details += ", ".join([p.perspective.value for p in processed_perspectives])
        
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
            return "I cannot provide a definitive answer due to conflicting information from different perspectives."
        else:
            return "I don't have sufficient information to answer this question confidently."


def run_stage_2(question: str, stage1_result: DiscoveryResult, model: str,
                k: int = 2, n: int = 4, theta_max: float = 0.5, rho_null: float = 0.3) -> Stage2Result:
    """
    Execute Stage 2: Perspective Resolution
    
    Args:
        stage1_result: Output from Stage 1 (dimension, perspectives, weights)
        question: The input question
        model: LLM model to use
        k: Number of chains of thought to generate per perspective
        n: Number of answers to sample from CoTs
        theta_max: Threshold for Type-1 abstention (knowledge contradiction)
        rho_null: Threshold for Type-2 abstention (knowledge insufficiency)
        
    Returns:
        Stage2Result with final answer and analysis
    """
    print("🚀 Starting Stage 2: Perspective Resolution")
    print(f"Question: {question}")
    print(f"Dimension: {stage1_result.best_dimension.name}")
    print(f"Perspectives: {[p.value for p in stage1_result.perspectives]}")
    print()
    
    processed_perspectives = []
    
    # Process each perspective
    for perspective in stage1_result.perspectives:
        print(f"Processing perspective: {perspective.value}")
        
        # Step 1: Generate K chains of thought
        chains_of_thought = generate_chains_of_thought(
            question, perspective, stage1_result.best_dimension.name, model, k
        )
        print(f"  Generated {len(chains_of_thought)} chains of thought")
        for i, cot in enumerate(chains_of_thought):
            print(f"    CoT {i+1}: {cot}")
        
        # Step 2: Sample N answers from chains
        reasonings = []
        samples_per_cot = n // k
        extra_samples = n % k
        
        for i, cot in enumerate(chains_of_thought):
            # Distribute samples across CoTs
            cot_samples = samples_per_cot + (1 if i < extra_samples else 0)
            
            for j in range(cot_samples):
                answer = generate_answer_from_cot(question, perspective, cot, model)
                
                # Assign probability (simplified - could use actual logprobs)
                probability = random.uniform(0.6, 0.9)
                
                reasoning = Reasoning(
                    cot_position=i,
                    answer=answer,
                    probability=probability
                )
                reasonings.append(reasoning)
                print(f"    Sample {len(reasonings)}: '{answer}' (p={probability:.3f}) from CoT {i+1}")
        
        print(f"  Generated {len(reasonings)} answer samples")
        
        # Step 3: Estimate causal effect using AIPW
        causal_effect = estimate_causal_effect_aipw(reasonings)
        print(f"  AIPW causal effect: {causal_effect:.4f}")
        
        # Create processed perspective
        proc_persp = ProcessedPerspective(
            perspective=perspective,
            chains_of_thought=chains_of_thought,
            reasonings=reasonings,
            causal_effect=causal_effect
        )
        processed_perspectives.append(proc_persp)
    
    # Perform CAD analysis for abstention policy
    print("\n🎯 Performing CAD Analysis...")
    cad_score, centroid_null_similarity, angular_deviations = compute_cad_analysis(processed_perspectives)
    
    print(f"CAD Score: {cad_score:.4f} (threshold: {theta_max})")
    print(f"Centroid-Null Similarity: {centroid_null_similarity:.4f}")
    print(f"Null Distance: {1 - centroid_null_similarity:.4f} (threshold: {rho_null})")
    
    # Three-way decision gate
    abstention_type = None
    
    if cad_score > theta_max:
        # Type-1 Abstention: Knowledge Contradiction
        print("🚫 Decision: TYPE-1 ABSTENTION (Knowledge Contradiction)")
        abstention_type = "TYPE1"
        final_answer = generate_abstention_response(question, "TYPE1", processed_perspectives, model)
        
    elif (1 - centroid_null_similarity) <= rho_null:
        # Type-2 Abstention: Knowledge Insufficiency
        print("🤷 Decision: TYPE-2 ABSTENTION (Knowledge Insufficiency)")
        abstention_type = "TYPE2"
        final_answer = generate_abstention_response(question, "TYPE2", processed_perspectives, model)
        
    else:
        # Aggregation
        print("✅ Decision: AGGREGATION")
        final_answer = aggregate_perspectives(question, processed_perspectives, model)
    
    print(f"\nFinal Answer: {final_answer}")
    
    # Create result
    result = Stage2Result(
        question=question,
        dimension=stage1_result.best_dimension.name,
        processed_perspectives=processed_perspectives,
        final_answer=final_answer,
        abstention_type=abstention_type,
        cad_score=cad_score,
        centroid_null_similarity=centroid_null_similarity
    )
    
    print("🏁 Stage 2 Complete!")
    return result
