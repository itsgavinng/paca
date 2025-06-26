"""
Base agent utilities for creating LLM instances based on model names.
"""
from langchain_openai import AzureChatOpenAI
from langchain_fireworks import ChatFireworks
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import os

load_dotenv()

class BaseAgent:
    """Base class for creating and managing LLM agents."""
    
    @staticmethod
    def get_llm(model: str, temperature: float = 0.0) -> Any:
        """
        Get the appropriate LLM based on the model name.
        
        Args:
            model: The model name/identifier
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            An LLM instance (AzureChatOpenAI, ChatGroq, or ChatFireworks)
            
        Raises:
            ValueError: If the model name is not recognized
        """
        if model.startswith("gpt"):
            # Map model names to deployment names
            deployment_name = "gpt-4o"  # Default deployment name
            
            return AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=deployment_name,
                temperature=temperature
            )
        else:
            # Fireworks models
            return ChatFireworks(
                api_key=os.getenv("FIREWORKS_API_KEY"),
                model=model,
                temperature=temperature
            )

    @staticmethod
    def create_agent(
        model: str,
        prompt_template: str,
        temperature: float = 0.0,
        prompt_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a generic agent with the specified model and prompt.
        
        Args:
            model: The model to use (e.g., "gpt-4", "llama2-70b-4096")
            prompt_template: The prompt template string
            temperature: Controls randomness (0.0 to 1.0)
            prompt_params: Optional parameters to format the prompt template
            
        Returns:
            An LLM instance configured with the prompt template
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableSequence
        
        # Create the prompt template
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Get the appropriate LLM
        llm = BaseAgent.get_llm(model, temperature)
        
        # Create and return the runnable sequence
        return prompt | llm

# Example usage:
if __name__ == "__main__":
    # Example prompt template
    example_prompt = """
    You are an AI assistant. Your role is to {role}.
    
    Context:
    {context}
    
    Your response:
    """
    
    # Example parameters
    params = {
        "role": "analyze the given text",
        "context": "This is a sample context"
    }
    
    # Create an agent with GPT-4
    agent = BaseAgent.create_agent(
        model="gpt-4",
        prompt_template=example_prompt,
        temperature=0.0
    )
    
    # Invoke the agent
    result = agent.invoke(params)
    print(result) 