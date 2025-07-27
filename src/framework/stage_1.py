"""
Stage 1: Dual-Agent Aspect Discovery

This module implements the main orchestration for Algorithm 1: Dual-Agent Aspect Discovery.

Algorithm Steps:
1. Dimension Discovery & Selection - Find the best context dimension X
2. Aspect Discovery - Find aspects {x_i} for the chosen X  
3. Weighting - Assign weights {w_i} to the aspects
"""

from typing import Dict
import llms.MODELS as MODELS
from framework.models import DiscoveryResult, Dimension, Aspect
from framework.agents import DiscoveryAgent, CriticalAgent


class DualAgentAspectDiscovery:
    """
    Main orchestrator for the Dual-Agent Aspect Discovery algorithm.
    
    This class coordinates the interaction between Discovery Agent and Critical Agent through
    multiple debate rounds to discover the best dimension, aspects, and weights.
    """
    
    def __init__(self, 
                 dagent_model: str = MODELS.OPENAI_GPT41,
                 cagent_model: str = MODELS.OPENAI_GPT41,
                 debate_rounds: int = 2,
                 weight_threshold: float = 0.1,
                 x_max: int = 5):
        """
        Initialize the dual-agent system.
        
        Args:
            dagent_model: Model for the Discovery Agent (DAgent)
            cagent_model: Model for the Critical Agent (CAgent)  
            debate_rounds: Number of debate rounds T (default: 2)
            weight_threshold: Threshold for weight convergence
            x_max: Maximum number of aspects to discover (default: 5)
        """
        self.debate_rounds = debate_rounds
        self.weight_threshold = weight_threshold
        self.x_max = x_max
        
        # Initialize agents
        self.dagent = DiscoveryAgent(dagent_model)
        self.cagent = CriticalAgent(cagent_model)
    
    def run_algorithm(self, question: str) -> DiscoveryResult:
        """
        Execute Algorithm 1: Dual-Agent Aspect Discovery
        
        Args:
            question: The input question Q
            
        Returns:
            DiscoveryResult containing the best dimension, aspects, and final weights
        """
        print(f"🚀 Starting Dual-Agent Aspect Discovery")
        print(f"Question: {question}")
        print(f"Debate Rounds: {self.debate_rounds}")
        print()
        
        # Step 1: Dimension Discovery & Selection
        best_dimension = self._discover_best_dimension(question)
        
        # Step 2: Aspect Discovery
        current_aspects = self._discover_aspects(best_dimension, question)
        
        # Step 3: Weighting
        final_weights = self._assign_weights(current_aspects, best_dimension, question)
        
        # Update aspect objects with final weights
        for aspect in current_aspects:
            aspect.weight = final_weights.get(aspect.value, 0.0)
        
        # Return final result
        result = DiscoveryResult(
            best_dimension=best_dimension,
            aspects=current_aspects,
            final_weights=final_weights
        )
        
        print("\n" + "=" * 60)
        print("🏁 ALGORITHM COMPLETE")
        print("=" * 60)
        print(f"Dimension (X): {result.best_dimension.name}")
        print(f"Aspects ({{x_i}}): {[p.value for p in result.aspects]}")
        print(f"Weights ({{w_i}}): {list(result.final_weights.values())}")
        
        return result
    
    def _discover_best_dimension(self, question: str) -> Dimension:
        """Step 1: Discover and select the best dimension through debate rounds."""
        print("=" * 60)
        print("STEP 1: DIMENSION DISCOVERY & SELECTION")
        print("=" * 60)
        
        current_dimensions = []
        for round_num in range(self.debate_rounds):
            print(f"\n--- Round {round_num + 1}/{self.debate_rounds} ---")
            
            # Discovery Agent discovers and ranks dimensions
            print("🔍 Discovery Agent discovering dimensions...")
            current_dimensions = self.dagent.discover_and_rank_dimensions(question)
            print(f"Discovery Agent proposed {len(current_dimensions)} dimensions")
            for i, dim in enumerate(current_dimensions[:3]):  # Show top 3
                print(f"  {i+1}. {dim.name} (score: {dim.score:.2f})")
            
            # Critical Agent tests dimensions against criteria
            print("✅ Critical Agent testing dimensions...")
            current_dimensions = self.cagent.test_dimensions(current_dimensions, question)
            print(f"Critical Agent validated {len(current_dimensions)} dimensions")
            for i, dim in enumerate(current_dimensions[:3]):  # Show top 3
                print(f"  {i+1}. {dim.name} (score: {dim.score:.2f})")
        
        # Select best dimension (X ← D_best)
        if not current_dimensions:
            raise ValueError("No valid dimensions found")
        
        best_dimension = max(current_dimensions, key=lambda d: d.score)
        print(f"\n🎯 Selected Best Dimension: {best_dimension.name}")
        print(f"   Score: {best_dimension.score:.2f}")
        print(f"   Description: {best_dimension.description}")
        
        return best_dimension
    
    def _discover_aspects(self, best_dimension: Dimension, question: str) -> list[Aspect]:
        """Step 2: Discover aspects within the chosen dimension through debate rounds."""
        print("\n" + "=" * 60)
        print("STEP 2: ASPECT DISCOVERY")
        print("=" * 60)
        
        current_aspects = []
        for round_num in range(self.debate_rounds):
            print(f"\n--- Round {round_num + 1}/{self.debate_rounds} ---")
            
            # Discovery Agent discovers aspects
            print("🔍 Discovery Agent discovering aspects...")
            current_aspects = self.dagent.discover_aspects(best_dimension, question, self.x_max)
            print(f"Discovery Agent proposed {len(current_aspects)} aspects")
            for i, aspect in enumerate(current_aspects):
                print(f"  {i+1}. {aspect.value}")
            
            # Critical Agent tests aspects
            print("✅ Critical Agent testing aspects...")
            current_aspects = self.cagent.test_aspects(current_aspects, best_dimension, question)
            print(f"Critical Agent validated {len(current_aspects)} aspects")
            
            # Limit to x_max aspects
            if len(current_aspects) > self.x_max:
                print(f"🔄 Limiting to top {self.x_max} aspects...")
                current_aspects = current_aspects[:self.x_max]
                print(f"   Kept {len(current_aspects)} aspects")
            
            for i, aspect in enumerate(current_aspects):
                print(f"  {i+1}. {aspect.value}")
        
        if not current_aspects:
            raise ValueError("No valid aspects found")
        
        return current_aspects
    
    def _assign_weights(self, current_aspects: list[Aspect], best_dimension: Dimension, question: str) -> Dict[str, float]:
        """Step 3: Assign weights to aspects through debate rounds."""
        print("\n" + "=" * 60)  
        print("STEP 3: WEIGHTING")
        print("=" * 60)
        
        dagent_weights = None
        cagent_weights = None
        
        for round_num in range(self.debate_rounds):
            print(f"\n--- Round {round_num + 1}/{self.debate_rounds} ---")
            
            # Discovery Agent assigns weights
            print("⚖️  Discovery Agent assigning weights...")
            dagent_weights = self.dagent.assign_weights(current_aspects, best_dimension, question)
            print("Discovery Agent weights:")
            for weight in dagent_weights.root:
                print(f"  {weight.value}: {weight.weight:.3f}")
            
            # Critical Agent assesses weights
            print("✅ Critical Agent assessing weights...")
            cagent_weights = self.cagent.assess_weights(dagent_weights, current_aspects, best_dimension, question)
            print("Critical Agent weights:")
            for weight in cagent_weights.root:
                print(f"  {weight.value}: {weight.weight:.3f}")
            
            # Check convergence
            weight_diff = self._calculate_weight_difference(dagent_weights, cagent_weights)
            print(f"Weight difference: {weight_diff:.4f} (threshold: {self.weight_threshold})")
            
            if weight_diff < self.weight_threshold:
                print("✅ Weights converged!")
                break
        else:
            print(f"⏰ Max rounds reached ({self.debate_rounds})")
        
        # Final weights: average of Discovery Agent and Critical Agent weights
        final_weights = self._average_weights(dagent_weights, cagent_weights, current_aspects)
        
        print("\n🎯 Final averaged weights:")
        for aspect_val, weight in final_weights.items():
            print(f"  {aspect_val}: {weight:.3f}")
        
        return final_weights
    
    def _calculate_weight_difference(self, weights_d, weights_c) -> float:
        """Calculate ||{w_i}_Discovery - {w_i}_Critical|| for convergence check."""
        d_weights = {w.value: w.weight for w in weights_d.root}
        c_weights = {w.value: w.weight for w in weights_c.root}
        
        all_aspects = set(d_weights.keys()) | set(c_weights.keys())
        
        diff_sum = 0.0
        for aspect in all_aspects:
            w_d = d_weights.get(aspect, 0.0)
            w_c = c_weights.get(aspect, 0.0)
            diff_sum += (w_d - w_c) ** 2
        
        return (diff_sum ** 0.5)  # L2 norm
    
    def _average_weights(self, weights_d, weights_c, aspects: list[Aspect]) -> Dict[str, float]:
        """Calculate avg({w_i}_Discovery, {w_i}_Critical) for final weights."""
        d_weights = {w.value: w.weight for w in weights_d.root}
        c_weights = {w.value: w.weight for w in weights_c.root}
        
        final_weights = {}
        
        for aspect in aspects:
            w_d = d_weights.get(aspect.value, 0.0)
            w_c = c_weights.get(aspect.value, 0.0)
            final_weights[aspect.value] = (w_d + w_c) / 2.0
        
        # Normalize to ensure sum = 1.0
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v/total for k, v in final_weights.items()}
        
        return final_weights


def run_stage_1(question: str, 
                dagent_model: str = MODELS.OPENAI_GPT41,
                cagent_model: str = MODELS.OPENAI_GPT41,
                debate_rounds: int = 2,
                weight_threshold: float = 0.1,
                x_max: int = 5) -> DiscoveryResult:
    """
    Main entry point for Stage 1: Dual-Agent Aspect Discovery
    
    Args:
        question: The input question to analyze
        dagent_model: Model for Discovery Agent  
        cagent_model: Model for Critical Agent
        debate_rounds: Number of debate rounds (T in algorithm, default: 2)
        weight_threshold: Convergence threshold for weights
        x_max: Maximum number of aspects to discover (default: 5)
        
    Returns:
        DiscoveryResult containing dimension, aspects, and weights
    """
    discovery = DualAgentAspectDiscovery(
        dagent_model=dagent_model,
        cagent_model=cagent_model,
        debate_rounds=debate_rounds,
        weight_threshold=weight_threshold,
        x_max=x_max
    )
    
    return discovery.run_algorithm(question) 