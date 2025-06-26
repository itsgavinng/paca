"""
Fireworks AI implementation for LLM chat.
"""
from typing import Optional, Dict, Tuple, List
import fireworks.client
import math
from .base import ChatResponse
from utils.maths import calc_nwgm

def chat(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ChatResponse:
    """
    Get a response from a Fireworks AI model.
    
    Args:
        prompt: The input prompt
        model: The model to use
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        ChatResponse object containing response, conversation history, logprobs, and nwgm
    """
    try:
        response = fireworks.client.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else 1000,
            logprobs=True,  # Enable log probabilities
            top_logprobs=1  # Get top 1 log probability for each token
        )
        
        # Extract the response text
        response_text = response.choices[0].text.strip()
        
        if ("Answer:\nAnswer:" in response_text or "Answer: Answer:" in response_text):
            raise Exception("Fireworks AI returned multiple Answer lines")
        
        # Create conversation history
        conversation_history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text}
        ]
        
        # Extract log probabilities
        logprobs = response.choices[0].logprobs if hasattr(response.choices[0], 'logprobs') else None
        
        if not logprobs:
            raise Exception("Log probabilities are null or missing from the API response")
        
        # Calculate nwgm (normalized weighted geometric mean) from logprobs
        extracted_logprobs = extract_token_logprobs(logprobs)
        if not extracted_logprobs:
            raise Exception("Failed to extract token log probabilities from the response")
        # print("Extracted logprobs:", extracted_logprobs)  # DEBUG
        nwgm = calc_nwgm(extracted_logprobs)
        
        return ChatResponse(
            response=response_text,
            conversation_history=conversation_history,
            logprobs=logprobs,
            nwgm=nwgm
        )
        
    except Exception as e:
        raise Exception(f"Error calling Fireworks AI: {str(e)}") 
    
    
def extract_token_logprobs(logprobs) -> List[float]:
    """
    Given a list of NewLogProbsContent objects, return a list of
    the chosen-token log probabilities for each token position.
    """
    token_logprobs: List[float] = []
    
    # Access the content list which contains the token information
    for content in logprobs.content:
        # Each content has a logprob attribute
        if hasattr(content, 'logprob'):
            token_logprobs.append(content.logprob)
    
    return token_logprobs