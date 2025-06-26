"""
Utility module for Azure OpenAI API integration.
"""
import os
from typing import Optional, List, Dict, Union, Tuple
from openai import AzureOpenAI
from dotenv import load_dotenv
from utils.maths import calc_nwgm
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

@dataclass
class ChatResponse:
    """Class to encapsulate chat response data."""
    response: str
    conversation_history: List[Dict[str, str]]
    logprobs: Optional[Dict]
    nwgm: Optional[float]

    def __str__(self) -> str:
        return self.response

class AzureOpenAIClient:
    """Client for interacting with Azure OpenAI API."""
    
    def __init__(self):
        """Initialize the Azure OpenAI client with environment variables."""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        
        if not all([self.api_key, self.endpoint]):
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables must be set")
        
        # Validate endpoint format
        if not self.endpoint.startswith("https://") or not self.endpoint.endswith(".openai.azure.com/"):
            raise ValueError("AZURE_OPENAI_ENDPOINT must be a valid Azure OpenAI endpoint URL (e.g., https://your-resource-name.openai.azure.com/)")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
    
    def get_completion(
        self,
        prompt: str,
        model: str,
        temperature: float = 0,
        max_completion_tokens: Optional[int] = None,
        system_message: str = "You are a helpful assistant.",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Optional[Dict]]:
        """
        Get a completion from Azure OpenAI API.
        
        Args:
            prompt: The input prompt
            model: The deployment name (not model name)
            temperature: Controls randomness (0.0 to 1.0)
            max_completion_tokens: Maximum number of tokens in the response
            system_message: Message to set the behavior of the assistant
            conversation_history: List of previous messages
            
        Returns:
            Tuple of (response_text, logprobs)
        """
        # Validate parameters
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        if max_completion_tokens is not None and max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        
        # Prepare messages
        messages = [{"role": "system", "content": system_message}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=model,  # This is actually the deployment name
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                logprobs=True,
                top_logprobs=5
            )
            
            response_text = response.choices[0].message.content
            logprobs = response.choices[0].logprobs if hasattr(response.choices[0], 'logprobs') else None
            
            return response_text, logprobs
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                raise Exception("API rate limit exceeded. Please try again later.")
            elif "timeout" in error_msg.lower():
                raise Exception("API request timed out. Please try again.")
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                raise Exception(f"Authentication failed. Please check your AZURE_OPENAI_API_KEY and endpoint. Error: {error_msg}")
            else:
                raise Exception(f"Error calling Azure OpenAI API: {error_msg}")

def chat(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_completion_tokens: Optional[int] = None,
    system_message: str = "You are a factuality checker assistant."
) -> ChatResponse:
    """
    Have a conversation with GPT, maintaining conversation history.
    
    Args:
        prompt: The user's message
        conversation_history: Previous conversation messages
        model: The deployment name (not model name)
        temperature: Controls randomness (0.0 to 1.0)
        max_completion_tokens: Maximum number of tokens in the response
        system_message: Message to set the behavior of the assistant
        
    Returns:
        ChatResponse object containing response, conversation history, logprobs, and nwgm
    """
    if conversation_history is None:
        conversation_history = []
    
    client = AzureOpenAIClient()
    response, logprobs = client.get_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        system_message=system_message,
        conversation_history=conversation_history
    )
    
    # Update conversation history
    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": response})
    
    # Extract logprobs
    if not logprobs:
        raise Exception("Log probabilities are null or missing from the API response")
    
    token_logprobs = None
    if hasattr(logprobs, 'content'):
        # Extract logprobs from ChoiceLogprobs object
        token_logprobs = [token.logprob for token in logprobs.content]
    elif isinstance(logprobs, dict) and 'content' in logprobs:
        # Handle dict format
        token_logprobs = [token['logprob'] for token in logprobs['content']]
    
    if not token_logprobs:
        raise Exception("Failed to extract token log probabilities from the response")

    nwgm = calc_nwgm(token_logprobs)
    
    return ChatResponse(
        response=response,
        conversation_history=conversation_history,
        logprobs=logprobs,
        nwgm=nwgm
    )