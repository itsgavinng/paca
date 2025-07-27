"""
Critical Agent

Implements the critical evaluation functionality for dimensions, aspects, and weights
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
    CAGENT_TEST_DIMENSIONS,
    CAGENT_TEST_PERSPECTIVES,
    CAGENT_ASSESS_WEIGHTS
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


class CriticalAgent:
    """
    Critical Agent that critically evaluates proposals from the Discovery Agent.
    
    The Critical Agent is responsible for quality control and validation against strict criteria.
    """
    
    def __init__(self, model: str):
        """
        Initialize the Criteria Agent.
        
        Args:
            model: LLM model identifier to use for this agent
        """
        self.model = model
        self.llm = BaseAgent.get_llm(model=model)
        
        # Initialize parsers
        self.dimension_parser = PydanticOutputParser(pydantic_object=DimensionList)
        self.aspect_parser = PydanticOutputParser(pydantic_object=AspectList)
        self.weight_parser = PydanticOutputParser(pydantic_object=AspectWeightList)
    
    def test_dimensions(self, dimensions: List[Dimension], question: str) -> List[Dimension]:
        """
        Critically evaluate proposed dimensions against strict causal validity criteria.
        
        Args:
            dimensions: List of dimensions to evaluate
            question: The input question for context
            
        Returns:
            List of validated and re-ranked dimensions
        """
        dimensions_json = json.dumps([dim.dict() for dim in dimensions], indent=2)
        
        prompt_template = PromptTemplate(
            input_variables=["question", "dimensions_json"],
            template=CAGENT_TEST_DIMENSIONS,
            partial_variables={"format_instructions": self.dimension_parser.get_format_instructions()}
        )
        
        chain = prompt_template | self.llm | self.dimension_parser
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimensions_json": dimensions_json
        })
        return result.root
    
    def test_aspects(self, aspects: List[Aspect], best_dimension: Dimension, question: str) -> List[Aspect]:
        """
        Critically evaluate proposed aspects against strict causal validity criteria.
        
        Args:
            aspects: List of aspects to evaluate
            best_dimension: The dimension these aspects belong to
            question: The input question for context
            
        Returns:
            List of validated aspects
        """
        aspects_json = json.dumps([aspect.dict() for aspect in aspects], indent=2)
        
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", "aspects_json"],
            template=CAGENT_TEST_PERSPECTIVES,
            partial_variables={"format_instructions": self.aspect_parser.get_format_instructions()}
        )
        
        chain = prompt_template | self.llm | self.aspect_parser
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "aspects_json": aspects_json
        })
        return result.root
    
    def assess_weights(self, weight_assignment: AspectWeightList, aspects: List[Aspect], 
                      best_dimension: Dimension, question: str) -> AspectWeightList:
        """
        Critically assess weight assignments based on evidence quality and factual foundation.
        
        Args:
            weight_assignment: Proposed weight assignments from DAgent
            aspects: List of aspects being weighted
            best_dimension: The dimension these aspects belong to
            question: The input question for context
            
        Returns:
            Revised AspectWeightList with critical assessment
        """
        aspects_weights_json = json.dumps([weight.dict() for weight in weight_assignment.root], indent=2)
        dagent_justifications = "\n".join([weight.justification for weight in weight_assignment.root])
        
        prompt_template = PromptTemplate(
            input_variables=["question", "dimension_name", "dimension_description", 
                           "aspects_weights_json", "dagent_justifications"],
            template=CAGENT_ASSESS_WEIGHTS,
            partial_variables={"format_instructions": self.weight_parser.get_format_instructions()}
        )
        
        chain = prompt_template | self.llm | self.weight_parser
        result = run_agent_with_retry(chain, {
            "question": question,
            "dimension_name": best_dimension.name,
            "dimension_description": best_dimension.description,
            "aspects_weights_json": aspects_weights_json,
            "dagent_justifications": dagent_justifications
        })
        return result 