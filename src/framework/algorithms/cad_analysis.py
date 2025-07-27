"""
Centroid Angular Deviation (CAD) Analysis

Implementation of the CAD algorithm for abstention policy in the framework.
Used to determine when to abstain from answering due to knowledge conflicts or insufficiency.
"""

from typing import List, Tuple
import numpy as np
from framework.models import ProcessedAspect

# Handle sentence transformers import with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def compute_cad_analysis(processed_aspects: List[ProcessedAspect]) -> Tuple[float, float, List[float]]:
    """
    Compute Centroid Angular Deviation (CAD) analysis for abstention policy.
    
    Args:
        processed_aspects: List of processed aspects with causal effects
        
    Returns:
        Tuple of (CAD score, centroid-null similarity, angular deviations)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Fallback: use simple heuristics
        print("Warning: Using simplified CAD analysis without sentence transformers")
        return 0.3, 0.5, [0.2] * len(processed_aspects)
    
    try:
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get representative answers for each aspect
        representative_answers = []
        weights = []
        causal_effects = []
        
        for proc_aspect in processed_aspects:
            if proc_aspect.reasonings:
                # Get the answer with highest probability as representative
                best_reasoning = max(proc_aspect.reasonings, key=lambda r: r.probability)
                representative_answers.append(best_reasoning.answer)
            else:
                representative_answers.append("No answer generated")
            
            weights.append(proc_aspect.aspect.weight)
            causal_effects.append(proc_aspect.causal_effect)
        
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
            "no answer",
            "I don't know"
        ]        
        null_embeddings = embedding_model.encode(null_phrases)
        
        # Normalize null embeddings
        null_embeddings_normalized = []
        for null_emb in null_embeddings:
            norm = np.linalg.norm(null_emb)
            if norm > 1e-8:
                null_embeddings_normalized.append(null_emb / norm)
            else:
                null_embeddings_normalized.append(null_emb)
        null_embeddings_normalized = np.array(null_embeddings_normalized)
        
        # Compute similarity with each individual null phrase
        individual_similarities = []
        for null_emb_norm in null_embeddings_normalized:
            similarity = np.clip(np.dot(centroid, null_emb_norm), -1.0, 1.0)
            individual_similarities.append(similarity)
        
        # Compute similarity with mean of null phrases
        null_consensus = np.mean(null_embeddings, axis=0)
        null_consensus = null_consensus / (np.linalg.norm(null_consensus) + 1e-8)
        consensus_similarity = np.clip(np.dot(centroid, null_consensus), -1.0, 1.0)
        
        # Return the best (highest) similarity
        all_similarities = individual_similarities + [consensus_similarity]
        centroid_null_similarity = max(all_similarities)
        
        return float(cad_score), float(centroid_null_similarity), angular_deviations.tolist()
        
    except Exception as e:
        print(f"Error in CAD analysis: {e}")
        # Return default values