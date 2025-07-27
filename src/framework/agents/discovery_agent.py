"""
Discovery Agent

Implements the discovery functionality for dimensions, aspects, and weight assignment
in the dual-agent aspect discovery algorithm.
"""

from typing import List
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from llms.base_agent import BaseAgent
from framework.models import (
    Dimension, Aspect, DimensionList, AspectList, AspectWeightList
)
from utils.prompts import (
    DAGENT_DISCOVER_DIMENSIONS,
    DAGENT_DISCOVER_PERSPECTIVES,
    DAGENT_ASSIGN_WEIGHTS
)


def run_agent_with_retry(chain, inputs, max_retries=3):
    """Run a LangChain chain with retry logic."""
    for retry in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"❌ Attempt {retry + 1} failed: {str(e)}")
            if retry == max_retries - 1:
                raise e
            print(f"⏳ Waiting 2 seconds before retry...")
            import time
            time.sleep(2)
            continue


class DiscoveryAgent:
    """
    Discovery Agent that proposes dimensions, aspects, and weights.
    
    The Discovery Agent is responsible for the creative/generative aspects of the discovery process.
    """
    
    def __init__(self, model: str):
        """
        Initialize the Discovery Agent.
        
        Args:
            model: LLM model identifier to use for this agent
        """
        self.model = model
        self.llm = BaseAgent.get_llm(model=model)
        
        # Initialize parsers
        self.dimension_parser = PydanticOutputParser(pydantic_object=DimensionList)
        self.aspect_parser = PydanticOutputParser(pydantic_object=AspectList)
        self.weight_parser = PydanticOutputParser(pydantic_object=AspectWeightList)
    
    def discover_and_rank_dimensions(self, question: str) -> List[Dimension]:
        """
        Discover and rank potential context dimensions for the given question.
        
        Args:
            question: The input question to analyze
            
        Returns:
            List of discovered dimensions ranked by importance
        """
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=DAGENT_DISCOVER_DIMENSIONS,
            partial_variables={"format_instructions": self.dimension_parser.get_format_instructions()}
        )
        
        chain = prompt_template | self.llm | self.dimension_parser
        result = run_agent_with_retry(chain, {"question": question})
        return result.root
    
    def discover_aspects(self, best_dimension: Dimension, question: str, max_aspects: int = 5) -> List[Aspect]:
        """
        Discover aspects within the chosen dimension.
        
        Args:
            best_dimension: The selected dimension to explore
            question: The input question
            max_aspects: Maximum number of aspects to discover
            
        Returns:
            List of discovered aspects
        """
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", "dimension_justification", "max_aspects"],
            template=DAGENT_DISCOVER_PERSPECTIVES,
            partial_variables={"format_instructions": self.aspect_parser.get_format_instructions()}
        )
        
        chain = prompt_template | self.llm | self.aspect_parser
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "dimension_justification": best_dimension.justification,
            "max_aspects": max_aspects
        })
        return result.root
    
    def assign_weights(self, aspects: List[Aspect], best_dimension: Dimension, question: str) -> AspectWeightList:
        """
        Assign importance weights to aspects based on evidence quality.
        
        Args:
            aspects: List of aspects to weight
            best_dimension: The dimension these aspects belong to
            question: The input question
            
        Returns:
            AspectWeightList with assigned weights and justifications
        """
        aspects_json = json.dumps([aspect.dict() for aspect in aspects], indent=2)
        
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", "perspectives_json"],
            template=DAGENT_ASSIGN_WEIGHTS,
            partial_variables={"format_instructions": self.weight_parser.get_format_instructions()}
        )
        
        chain = prompt_template | self.llm | self.weight_parser
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "perspectives_json": aspects_json
        })
        return result 