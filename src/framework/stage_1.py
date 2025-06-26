"""
Stage 1: Dual-Agent Perspective Discovery

This module implements Algorithm 1: Dual-Agent Perspective Discovery
following the exact structure provided in the algorithm screenshot.

The algorithm has 3 main steps:
1. Dimension Discovery & Selection - Find the best context dimension X
2. Perspective Discovery - Find perspectives {x_i} for the chosen X  
3. Weighting - Assign weights {w_i} to the perspectives

Each step involves debate rounds between DAgent (Discovery Agent) and CAgent (Criteria Agent).
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, RootModel
import json
from llms.base_agent import BaseAgent
import llms.MODELS as MODELS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from utils.prompts import (
    DAGENT_DISCOVER_DIMENSIONS,
    DAGENT_DISCOVER_PERSPECTIVES, 
    DAGENT_ASSIGN_WEIGHTS,
    CAGENT_TEST_DIMENSIONS,
    CAGENT_TEST_PERSPECTIVES,
    CAGENT_ASSESS_WEIGHTS
)


class Dimension(BaseModel):
    """Represents a discovered dimension/context variable."""
    name: str
    description: str
    justification: str
    score: float = 0.0
    
    
class Perspective(BaseModel):
    """Represents a perspective/stratum within a dimension."""
    value: str
    description: str
    justification: str
    weight: float = 0.0


class WeightAssignment(BaseModel):
    """Represents weight assignments from an agent."""
    perspective_weights: Dict[str, float]
    justification: str


class DiscoveryResult(BaseModel):
    """Final result of the perspective discovery process."""
    best_dimension: Dimension
    perspectives: List[Perspective]
    final_weights: Dict[str, float]


# Root models for LangChain parsing
class DimensionList(RootModel):
    """Root model for parsing list of dimensions."""
    root: List[Dimension]


class PerspectiveList(RootModel):
    """Root model for parsing list of perspectives."""
    root: List[Perspective]


def run_agent_with_retry(chain, inputs, max_retries=3):
    """Run a LangChain chain with retry logic."""
    for retry in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if retry == max_retries - 1:
                raise e
            print(f"Attempt {retry + 1} failed, retrying...")
            continue


class DualAgentPerspectiveDiscovery:
    """
    Implementation of Algorithm 1: Dual-Agent Perspective Discovery
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
            cagent_model: Model for the Criteria Agent (CAgent)  
            debate_rounds: Number of debate rounds T (default: 2 as per algorithm)
            weight_threshold: Threshold for weight convergence in Step 3
            x_max: Maximum number of perspectives to discover (default: 5)
        """
        self.dagent_model = dagent_model
        self.cagent_model = cagent_model
        self.debate_rounds = debate_rounds
        self.weight_threshold = weight_threshold
        self.x_max = x_max
        
        # Initialize LLMs for LangChain using BaseAgent
        self.dagent_llm = BaseAgent.get_llm(model=dagent_model)
        self.cagent_llm = BaseAgent.get_llm(model=cagent_model)
        
        # Initialize parsers
        self.dimension_parser = PydanticOutputParser(pydantic_object=DimensionList)
        self.perspective_parser = PydanticOutputParser(pydantic_object=PerspectiveList)
        self.weight_parser = PydanticOutputParser(pydantic_object=WeightAssignment)
    
    def discover_and_rank(self, question: str) -> List[Dimension]:
        """
        DAgent.discover_and_rank(Q) - Step 1 of the algorithm.
        Discovery Agent discovers and ranks potential dimensions.
        """
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=DAGENT_DISCOVER_DIMENSIONS,
            partial_variables={"format_instructions": self.dimension_parser.get_format_instructions()}
        )
        
        # Create chain
        chain = prompt_template | self.dagent_llm | self.dimension_parser
        
        # Execute with retry
        result = run_agent_with_retry(chain, {"question": question})
        return result.root
    
    def test_dimensions(self, dimensions: List[Dimension], question: str) -> List[Dimension]:
        """
        CAgent.test(D_ranked, C_dis) - Criteria Agent tests dimensions against criteria.
        """
        dimensions_json = json.dumps([d.dict() for d in dimensions], indent=2)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "dimensions_json"],
            template=CAGENT_TEST_DIMENSIONS,
            partial_variables={"format_instructions": self.dimension_parser.get_format_instructions()}
        )
        
        # Create chain
        chain = prompt_template | self.cagent_llm | self.dimension_parser
        
        # Execute with retry
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimensions_json": dimensions_json
        })
        return result.root
    
    def discover_perspectives(self, best_dimension: Dimension, question: str) -> List[Perspective]:
        """
        DAgent.discover_perspectives(X) - Step 2 of the algorithm.
        Discovery Agent discovers perspectives for the chosen dimension.
        """
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", "dimension_justification"],
            template=DAGENT_DISCOVER_PERSPECTIVES,
            partial_variables={"format_instructions": self.perspective_parser.get_format_instructions()}
        )
        
        # Create chain
        chain = prompt_template | self.dagent_llm | self.perspective_parser
        
        # Execute with retry
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "dimension_justification": best_dimension.justification
        })
        return result.root
    
    def test_perspectives(self, perspectives: List[Perspective], best_dimension: Dimension, question: str) -> List[Perspective]:
        """
        CAgent.test({x_i}, C_dis) - Criteria Agent tests perspectives against criteria.
        """
        perspectives_json = json.dumps([p.dict() for p in perspectives], indent=2)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", "perspectives_json"],
            template=CAGENT_TEST_PERSPECTIVES,
            partial_variables={"format_instructions": self.perspective_parser.get_format_instructions()}
        )
        
        # Create chain
        chain = prompt_template | self.cagent_llm | self.perspective_parser
        
        # Execute with retry
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "perspectives_json": perspectives_json
        })
        return result.root
    
    def assign_weights(self, perspectives: List[Perspective], best_dimension: Dimension, question: str) -> WeightAssignment:
        """
        DAgent.assign_weights({x_i}) - Discovery Agent assigns weights to perspectives.
        """
        perspectives_json = json.dumps([p.dict() for p in perspectives], indent=2)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", "perspectives_json"],
            template=DAGENT_ASSIGN_WEIGHTS,
            partial_variables={"format_instructions": self.weight_parser.get_format_instructions()}
        )
        
        # Create chain
        chain = prompt_template | self.dagent_llm | self.weight_parser
        
        # Execute with retry
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "perspectives_json": perspectives_json
        })
        return result
    
    def assess_weights(self, weight_assignment: WeightAssignment, perspectives: List[Perspective], 
                      best_dimension: Dimension, question: str) -> WeightAssignment:
        """
        CAgent.assess({w_i}_D) - Criteria Agent assesses the weights from DAgent.
        """
        perspective_weights_json = json.dumps(
            dict(zip([p.value for p in perspectives], 
                    [weight_assignment.perspective_weights.get(p.value, 0.0) for p in perspectives])), 
            indent=2
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", 
                           "perspective_weights_json", "dagent_justification"],
            template=CAGENT_ASSESS_WEIGHTS,
            partial_variables={"format_instructions": self.weight_parser.get_format_instructions()}
        )
        
        # Create chain
        chain = prompt_template | self.cagent_llm | self.weight_parser
        
        # Execute with retry
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "perspective_weights_json": perspective_weights_json,
            "dagent_justification": weight_assignment.justification
        })
        return result
    
    def run_algorithm(self, question: str) -> DiscoveryResult:
        """
        Execute Algorithm 1: Dual-Agent Perspective Discovery
        
        Args:
            question: The input question Q
            
        Returns:
            DiscoveryResult containing the best dimension, perspectives, and final weights
        """
        print(f"🚀 Starting Dual-Agent Perspective Discovery")
        print(f"Question: {question}")
        print(f"Debate Rounds: {self.debate_rounds}")
        print()
        
        # Step 1: Dimension Discovery & Selection
        print("=" * 60)
        print("STEP 1: DIMENSION DISCOVERY & SELECTION")
        print("=" * 60)
        
        current_dimensions = []
        for round_num in range(self.debate_rounds):
            print(f"\n--- Round {round_num + 1}/{self.debate_rounds} ---")
            
            # DAgent discovers and ranks dimensions
            print("🔍 DAgent discovering dimensions...")
            current_dimensions = self.discover_and_rank(question)
            print(f"DAgent proposed {len(current_dimensions)} dimensions")
            for i, dim in enumerate(current_dimensions[:3]):  # Show top 3
                print(f"  {i+1}. {dim.name} (score: {dim.score:.2f})")
            
            # CAgent tests dimensions against criteria
            print("✅ CAgent testing dimensions...")
            current_dimensions = self.test_dimensions(current_dimensions, question)
            print(f"CAgent validated {len(current_dimensions)} dimensions")
            for i, dim in enumerate(current_dimensions[:3]):  # Show top 3
                print(f"  {i+1}. {dim.name} (score: {dim.score:.2f})")
        
        # Select best dimension (X ← D_best)
        if not current_dimensions:
            raise ValueError("No valid dimensions found")
        
        best_dimension = max(current_dimensions, key=lambda d: d.score)
        print(f"\n🎯 Selected Best Dimension: {best_dimension.name}")
        print(f"   Score: {best_dimension.score:.2f}")
        print(f"   Description: {best_dimension.description}")
        
        # Step 2: Perspective Discovery  
        print("\n" + "=" * 60)
        print("STEP 2: PERSPECTIVE DISCOVERY")
        print("=" * 60)
        
        current_perspectives = []
        for round_num in range(self.debate_rounds):
            print(f"\n--- Round {round_num + 1}/{self.debate_rounds} ---")
            
            # DAgent discovers perspectives
            print("🔍 DAgent discovering perspectives...")
            current_perspectives = self.discover_perspectives(best_dimension, question)
            print(f"DAgent proposed {len(current_perspectives)} perspectives")
            for i, persp in enumerate(current_perspectives):
                print(f"  {i+1}. {persp.value}")
            
            # CAgent tests perspectives
            print("✅ CAgent testing perspectives...")
            current_perspectives = self.test_perspectives(current_perspectives, best_dimension, question)
            print(f"CAgent validated {len(current_perspectives)} perspectives")
            
            # Limit to x_max perspectives (keep top-weighted ones if weights exist, otherwise keep first x_max)
            if len(current_perspectives) > self.x_max:
                print(f"🔄 Limiting to top {self.x_max} perspectives...")
                # Sort by weight if available, otherwise keep original order
                if hasattr(current_perspectives[0], 'weight') and current_perspectives[0].weight > 0:
                    current_perspectives = sorted(current_perspectives, key=lambda p: p.weight, reverse=True)[:self.x_max]
                else:
                    current_perspectives = current_perspectives[:self.x_max]
                print(f"   Kept {len(current_perspectives)} perspectives")
            
            for i, persp in enumerate(current_perspectives):
                print(f"  {i+1}. {persp.value}")
        
        if not current_perspectives:
            raise ValueError("No valid perspectives found")
        
        # Step 3: Weighting
        print("\n" + "=" * 60)  
        print("STEP 3: WEIGHTING")
        print("=" * 60)
        
        dagent_weights = None
        cagent_weights = None
        
        for round_num in range(self.debate_rounds):
            print(f"\n--- Round {round_num + 1}/{self.debate_rounds} ---")
            
            # DAgent assigns weights
            print("⚖️  DAgent assigning weights...")
            dagent_weights = self.assign_weights(current_perspectives, best_dimension, question)
            print("DAgent weights:")
            for persp_val, weight in dagent_weights.perspective_weights.items():
                print(f"  {persp_val}: {weight:.3f}")
            
            # CAgent assesses weights
            print("✅ CAgent assessing weights...")
            cagent_weights = self.assess_weights(dagent_weights, current_perspectives, best_dimension, question)
            print("CAgent weights:")
            for persp_val, weight in cagent_weights.perspective_weights.items():
                print(f"  {persp_val}: {weight:.3f}")
            
            # Check convergence: ||{w_i}_D - {w_i}_C|| < threshold
            weight_diff = self._calculate_weight_difference(dagent_weights, cagent_weights)
            print(f"Weight difference: {weight_diff:.4f} (threshold: {self.weight_threshold})")
            
            if weight_diff < self.weight_threshold:
                print("✅ Weights converged!")
                break
        else:
            print(f"⏰ Max rounds reached ({self.debate_rounds})")
        
        # Final weights: {w_i} ← avg({w_i}_D, {w_i}_C)
        final_weights = self._average_weights(dagent_weights, cagent_weights, current_perspectives)
        
        print("\n🎯 Final averaged weights:")
        for persp_val, weight in final_weights.items():
            print(f"  {persp_val}: {weight:.3f}")
        
        # Update perspective objects with final weights
        for persp in current_perspectives:
            persp.weight = final_weights.get(persp.value, 0.0)
        
        # Return X, {x_i}, {w_i}
        result = DiscoveryResult(
            best_dimension=best_dimension,
            perspectives=current_perspectives,
            final_weights=final_weights
        )
        
        print("\n" + "=" * 60)
        print("🏁 ALGORITHM COMPLETE")
        print("=" * 60)
        print(f"Dimension (X): {result.best_dimension.name}")
        print(f"Perspectives ({{x_i}}): {[p.value for p in result.perspectives]}")
        print(f"Weights ({{w_i}}): {list(result.final_weights.values())}")
        
        return result
    

    
    def _calculate_weight_difference(self, weights_d: WeightAssignment, weights_c: WeightAssignment) -> float:
        """Calculate ||{w_i}_D - {w_i}_C|| for convergence check."""
        all_perspectives = set(weights_d.perspective_weights.keys()) | set(weights_c.perspective_weights.keys())
        
        diff_sum = 0.0
        for persp in all_perspectives:
            w_d = weights_d.perspective_weights.get(persp, 0.0)
            w_c = weights_c.perspective_weights.get(persp, 0.0)
            diff_sum += (w_d - w_c) ** 2
        
        return (diff_sum ** 0.5)  # L2 norm
    
    def _average_weights(self, weights_d: WeightAssignment, weights_c: WeightAssignment, 
                        perspectives: List[Perspective]) -> Dict[str, float]:
        """Calculate avg({w_i}_D, {w_i}_C) for final weights."""
        final_weights = {}
        
        for persp in perspectives:
            w_d = weights_d.perspective_weights.get(persp.value, 0.0)
            w_c = weights_c.perspective_weights.get(persp.value, 0.0)
            final_weights[persp.value] = (w_d + w_c) / 2.0
        
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
    Main entry point for Stage 1: Dual-Agent Perspective Discovery
    
    Args:
        question: The input question to analyze
        dagent_model: Model for Discovery Agent  
        cagent_model: Model for Criteria Agent
        debate_rounds: Number of debate rounds (T in algorithm, default: 2)
        weight_threshold: Convergence threshold for weights
        x_max: Maximum number of perspectives to discover (default: 5)
        
    Returns:
        DiscoveryResult containing dimension, perspectives, and weights
    """
    discovery = DualAgentPerspectiveDiscovery(
        dagent_model=dagent_model,
        cagent_model=cagent_model,
        debate_rounds=debate_rounds,
        weight_threshold=weight_threshold,
        x_max=x_max
    )
    
    return discovery.run_algorithm(question)

