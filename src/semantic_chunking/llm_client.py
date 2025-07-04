import logging
import re
import json
from typing import Optional, Dict

import anthropic

from .config import ANTHROPIC_API_KEY, LLM_MODEL

# Initialize the Anthropic client
try:
    CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
except anthropic.APIError as e:
    logging.error(f"Failed to initialize Anthropic client: {e}")
    CLIENT = None

def call_llm(prompt: str, system_prompt: str) -> Optional[str]:
    """
    Makes a call to the Anthropic API and handles potential errors.
    Returns the response text or None if an error occurs.
    """
    if not CLIENT:
        logging.error("Anthropic client is not initialized. Cannot make API call.")
        return None
        
    try:
        # Add a log message before every single API call
        logging.info(f"Making LLM call. System prompt: '{system_prompt[:50]}...'. User prompt length: {len(prompt)} chars.")
        
        message = CLIENT.messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            max_tokens=10000, # Sonnet 4 can go up to 64k tokens
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text
        logging.info(f"LLM call successful. Response length: {len(response_text)}")
        return response_text
    except anthropic.APIError as e:
        logging.error(f"Anthropic API Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM call: {e}")
    return None

def extract_json_from_llm_response(response: str) -> Optional[Dict]:
    """
    Extracts a JSON object from the LLM's response string.
    """
    if not response:
        return None
        
    try:
        # Find the first '{' and the last '}' to extract the JSON object
        json_start = response.find("{")
        json_end = response.rfind("}")
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response[json_start:json_end + 1]
            return json.loads(json_str)
        
        logging.warning(f"Could not find a valid JSON object in the LLM response. Response start: '{response[:200]}...' Response end: '...{response[-200:]}'")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from response: {e}\nResponse snippet:\n{response[:500]}")
    except Exception as e:
        logging.error(f"Error extracting JSON: {e}")
    
    return None 