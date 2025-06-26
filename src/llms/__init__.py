"""
Unified interface for LLM models.
"""
from typing import Optional, Dict, Tuple
from .azure_openai import chat as azure_chat
from .fireworks_ai import chat as fireworks_chat
from .base import ChatResponse
import time

def chat(
    prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: Optional[int] = 2000,
    provider: Optional[str] = "fireworks"
) -> ChatResponse:
    """
    Get a response from an LLM model.
    
    Args:
        prompt: The input prompt
        model: The model to use
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens in the response
        provider: Optional provider override ("together" for Together.ai, "fireworks" for Fireworks AI)
        
    Returns:
        ChatResponse object containing response, conversation history, logprobs, and nwgm
    """
    max_retries = 5
    delay = 1  # seconds
    last_exception = None
    for attempt in range(max_retries):
        try:
            # Route to Azure OpenAI for GPT models
            if model.startswith("gpt"):
                azure_max_tokens = max_tokens if max_tokens is not None else 1000
                return azure_chat(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_completion_tokens=azure_max_tokens
                )
            # Route to Fireworks AI for models containing "fireworks"
            else:
                return fireworks_chat(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
          
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise last_exception