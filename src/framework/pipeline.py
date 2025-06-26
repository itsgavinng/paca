"""
Pipeline for running the complete framework: Stage 1 + Stage 2

This module loads the dataset and processes each question through both stages
with retry logic for robust execution.
"""

import json
import time
from typing import List, Dict, Any
from pathlib import Path

from framework.stage_1 import run_stage_1
from framework.stage_2 import run_stage_2
import llms.MODELS as MODELS


def load_dataset(dataset_path: str = "datasets/sample_dataset.json") -> List[Dict[str, Any]]:
    """
    Load the dataset from JSON file.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        List of question dictionaries
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"📁 Loaded dataset with {len(dataset)} questions from {dataset_path}")
        return dataset
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON dataset: {e}")
        raise


def process_question_with_retry(question_data: Dict[str, Any], model: str, max_retries: int = 3, t: int = 2, k: int = 2, n: int = 4) -> Dict[str, Any]:
    """
    Process a single question through both stages with retry logic.
    
    Args:
        question_data: Dictionary containing question, answer, and answerable fields
        model: Model to use for processing
        max_retries: Maximum number of retry attempts
        t: Number of debate rounds for Stage 1
        k: Number of chains of thought per perspective for Stage 2
        n: Number of answer samples for Stage 2
        
    Returns:
        Dictionary containing results from both stages
    """
    question = question_data["question"]
    correct_answer = question_data["answer"]
    answerable = question_data["answerable"]
    
    print(f"\n{'='*80}")
    print(f"📝 Question: {question}")
    print(f"✅ Correct Answer: {correct_answer}")
    print(f"🎯 Answerable: {answerable}")
    print(f"{'='*80}")
    
    for attempt in range(max_retries):
        try:
            print(f"\n🚀 Attempt {attempt + 1}/{max_retries}")
            
            # Stage 1: Perspective Discovery
            print("\n" + "🔍 STAGE 1: PERSPECTIVE DISCOVERY")
            stage1_result = run_stage_1(
                question=question,
                dagent_model=model,
                cagent_model=model,
                debate_rounds=t,
                weight_threshold=0.1
            )
            
            print(f"✅ Stage 1 Complete - Dimension: {stage1_result.best_dimension.name}")
            print(f"   Perspectives: {[p.value for p in stage1_result.perspectives]}")
            
            # Stage 2: Perspective Resolution
            print("\n" + "🎯 STAGE 2: PERSPECTIVE RESOLUTION")
            stage2_result = run_stage_2(
                question=question,
                stage1_result=stage1_result,
                model=model,
                k=k,  # chains of thought per perspective
                n=n,  # answer samples
                theta_max=0.5,  # Type-1 abstention threshold
                rho_null=0.3   # Type-2 abstention threshold
            )

            
            # Prepare result summary
            result = {
                "question": question,
                "correct_answer": correct_answer,
                "answerable": answerable,
                "stage1_result": {
                    "dimension": stage1_result.best_dimension.name,
                    "perspectives": [p.value for p in stage1_result.perspectives],
                    "weights": list(stage1_result.final_weights.values())
                },
                "stage2_result": {
                    "final_answer": stage2_result.final_answer,
                    "abstention_type": stage2_result.abstention_type,
                    "cad_score": stage2_result.cad_score,
                    "centroid_null_similarity": stage2_result.centroid_null_similarity
                },
                "success": True,
                "attempts": attempt + 1
            }
            
            print(f"🎉 Successfully processed question in {attempt + 1} attempt(s)")
            return result
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"💥 All {max_retries} attempts failed for question: {question}")
                return {
                    "question": question,
                    "correct_answer": correct_answer,
                    "answerable": answerable,
                    "error": str(e),
                    "success": False,
                    "attempts": max_retries
                }


def run_pipeline(dataset_path: str = "datasets/sample_dataset.json",
                model: str = MODELS.OPENAI_GPT41,
                max_retries: int = 3,
                start_index: int = 0,
                end_index: int = None,
                t: int = 2,
                k: int = 2,
                n: int = 4) -> List[Dict[str, Any]]:
    """
    Run the complete pipeline on the dataset.
    
    Args:
        dataset_path: Path to the dataset file
        model: Model to use for both stages
        max_retries: Maximum retry attempts per question
        start_index: Starting index in dataset (for resuming)
        end_index: Ending index in dataset (None for all)
        t: Number of debate rounds for Stage 1
        k: Number of chains of thought per perspective for Stage 2
        n: Number of answer samples for Stage 2
        
    Returns:
        List of results for each processed question
    """
    print("🏁 Starting Framework Pipeline")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🤖 Model: {model}")
    print(f"🔄 Max Retries: {max_retries}")
    print(f"⚙️  Parameters: t={t}, k={k}, n={n}")
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Determine processing range
    if end_index is None:
        end_index = len(dataset)
    
    subset = dataset[start_index:end_index]
    print(f"📋 Processing questions {start_index} to {end_index-1} ({len(subset)} questions)")
    
    results = []
    successful = 0
    failed = 0
    
    for i, question_data in enumerate(subset):
        global_index = start_index + i
        print(f"\n{'🔥' * 20} PROCESSING QUESTION {global_index + 1}/{len(dataset)} {'🔥' * 20}")
        
        result = process_question_with_retry(question_data, model, max_retries, t, k, n)
        results.append(result)
        
        if result["success"]:
            successful += 1
        else:
            failed += 1
        
        print(f"\n📊 Progress: {successful} successful, {failed} failed, {len(results)} total")
    
    # Final summary
    print(f"\n{'🏆' * 50}")
    print("PIPELINE COMPLETE")
    print(f"{'🏆' * 50}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    print(f"📈 Success Rate: {successful/len(results)*100:.1f}%")
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: str = "results/pipeline_results.json"):
    """
    Save pipeline results to JSON file.
    
    Args:
        results: List of results from pipeline
        output_path: Path to save results
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to {output_path}")

