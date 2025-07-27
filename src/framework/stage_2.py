"""
Stage 2: Aspect Resolution

This module implements the main orchestration for Stage 2: Aspect Resolution.

Key components:
1. Chain of Thought Generation per aspect
2. Causal Effect Estimation using AIPW
3. Abstention Policy using CAD analysis
4. Three-way decision gate: Type-1 Abstention, Type-2 Abstention, or Aggregation
"""

import random
from framework.models import DiscoveryResult, Stage2Result, ProcessedAspect, Reasoning
from framework.processors import generate_chains_of_thought, generate_answer_from_cot
from framework.algorithms import (
    estimate_causal_effect_aipw, 
    compute_cad_analysis,
    aggregate_aspects,
    generate_abstention_response
)


def run_stage_2(question: str, stage1_result: DiscoveryResult, model: str,
                k: int = 2, n: int = 4, theta_max: float = 0.5, rho_null: float = 0.5) -> Stage2Result:
    """
    Execute Stage 2: Aspect Resolution
    
    Args:
        question: The input question
        stage1_result: Output from Stage 1 (dimension, aspects, weights)
        model: LLM model to use
        k: Number of chains of thought to generate per aspect
        n: Number of answers to sample from CoTs
        theta_max: Threshold for Type-1 abstention (knowledge contradiction)
        rho_null: Threshold for Type-2 abstention (knowledge insufficiency)
        
    Returns:
        Stage2Result with final answer and analysis
    """
    print("🚀 Starting Stage 2: Aspect Resolution")
    print(f"Question: {question}")
    print(f"Dimension: {stage1_result.best_dimension.name}")
    print(f"Aspects: {[p.value for p in stage1_result.aspects]}")
    print()
    
    # Process each aspect through the reasoning pipeline
    processed_aspects = []
    
    for aspect in stage1_result.aspects:
        processed_aspect = _process_aspect(
            question=question,
            aspect=aspect,
            dimension=stage1_result.best_dimension.name,
            model=model,
            k=k,
            n=n
        )
        processed_aspects.append(processed_aspect)
    
    # Perform CAD analysis for abstention policy
    print("\n🎯 Performing CAD Analysis...")
    cad_score, centroid_null_similarity, angular_deviations = compute_cad_analysis(processed_aspects)
    
    print(f"CAD Score: {cad_score:.4f} (threshold: {theta_max})")
    print(f"Centroid-Null Similarity: {centroid_null_similarity:.4f}")
    print(f"Null Distance: {1 - centroid_null_similarity:.4f} (threshold: {rho_null})")
    
    # Three-way decision gate
    abstention_type, final_answer = _make_decision(
        question=question,
        processed_aspects=processed_aspects,
        cad_score=cad_score,
        centroid_null_similarity=centroid_null_similarity,
        theta_max=theta_max,
        rho_null=rho_null,
        model=model
    )
    
    print(f"\nFinal Answer: {final_answer}")
    
    # Create and return result
    result = Stage2Result(
        question=question,
        dimension=stage1_result.best_dimension.name,
        processed_aspects=processed_aspects,
        final_answer=final_answer,
        abstention_type=abstention_type,
        cad_score=cad_score,
        centroid_null_similarity=centroid_null_similarity
    )
    
    print("🏁 Stage 2 Complete!")
    return result


def _process_aspect(question: str, aspect, dimension: str, model: str, k: int, n: int) -> ProcessedAspect:
    """
    Process a single aspect through the reasoning pipeline.
    
    This includes:
    1. Generating K chains of thought
    2. Sampling N answers from chains
    3. Estimating causal effect using AIPW
    """
    print(f"Processing aspect: {aspect.value}")
    
    # Step 1: Generate K chains of thought
    chains_of_thought = generate_chains_of_thought(
        question, aspect, dimension, model, k
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
            answer = generate_answer_from_cot(question, aspect, cot, model)
            
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
    
    # Create processed aspect
    return ProcessedAspect(
        aspect=aspect,
        chains_of_thought=chains_of_thought,
        reasonings=reasonings,
        causal_effect=causal_effect
    )


def _make_decision(question: str, processed_aspects, cad_score: float, 
                   centroid_null_similarity: float, theta_max: float, rho_null: float, 
                   model: str) -> tuple[str, str]:
    """
    Make the three-way decision: Type-1 Abstention, Type-2 Abstention, or Aggregation.
    
    Returns:
        Tuple of (abstention_type, final_answer)
    """
    if cad_score > theta_max:
        # Type-1 Abstention: Knowledge Contradiction
        print("🚫 Decision: TYPE-1 ABSTENTION (Knowledge Contradiction)")
        abstention_type = "TYPE1"
        final_answer = generate_abstention_response(question, "TYPE1", processed_aspects, model)
        
    elif (1 - centroid_null_similarity) <= rho_null:
        # Type-2 Abstention: Knowledge Insufficiency
        print("🤷 Decision: TYPE-2 ABSTENTION (Knowledge Insufficiency)")
        abstention_type = "TYPE2"
        final_answer = generate_abstention_response(question, "TYPE2", processed_aspects, model)
        
    else:
        # Aggregation
        print("✅ Decision: AGGREGATION")
        abstention_type = None
        final_answer = aggregate_aspects(question, processed_aspects, model)
    
    return abstention_type, final_answer 