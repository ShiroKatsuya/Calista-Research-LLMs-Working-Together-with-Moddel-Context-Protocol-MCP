"""
Utility functions for the minions package.
"""

import re
import json
import functools
from typing import Dict, Any, Optional


@functools.lru_cache(maxsize=32)
def escape_newlines_in_strings(json_str: str) -> str:
    """
    Escape newlines in JSON strings to ensure proper parsing.
    Uses a regular expression to find string content and replace newlines.
    
    Args:
        json_str: JSON string to process
        
    Returns:
        Processed JSON string with escaped newlines in string values
    """
    def replace_newlines(match):
        # Replace literal newlines with \\n
        content = match.group(0)
        return content.replace('\n', '\\n')
    
    # This pattern matches content inside double quotes, handling escaped quotes
    pattern = r'"(?:[^"\\]|\\.)*"'
    return re.sub(pattern, replace_newlines, json_str)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text, handling common issues like triple backticks and newlines.
    
    Args:
        text: String that may contain JSON
        
    Returns:
        Extracted JSON as a dictionary
    """
    # Try to find JSON enclosed in triple backticks
    json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    
    # If found in triple backticks
    if json_matches:
        for json_str in json_matches:
            try:
                # Try to parse the JSON directly
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try cleaning and repairing
                clean_json = clean_json_string(json_str)
                try:
                    return json.loads(clean_json)
                except json.JSONDecodeError:
                    # If still fails, try aggressive repair
                    continue
    
    # If not found in triple backticks, try to find JSON using brackets
    curly_matches = re.findall(r'({[\s\S]*?})', text)
    
    for json_str in curly_matches:
        try:
            # Try to parse potential JSON
            json_obj = json.loads(json_str)
            # Check if it has expected keys - adjust these as needed
            if any(key in json_obj for key in ["message", "decision", "answer", "status"]):
                return json_obj
        except:
            continue
    
    # Last resort: aggressive JSON repair on the whole text
    return aggressive_json_repair(text)


def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string for better parsing.
    
    Args:
        json_str: Potentially invalid JSON string
        
    Returns:
        Cleaned JSON string
    """
    # Replace common issues
    clean_str = json_str.strip()
    
    # Remove trailing commas before closing brackets
    clean_str = re.sub(r',\s*}', '}', clean_str)
    clean_str = re.sub(r',\s*]', ']', clean_str)
    
    # Fix unquoted keys
    clean_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', clean_str)
    
    # Fix single quotes
    clean_str = re.sub(r"'([^']*)'", r'"\1"', clean_str)
    
    # Escape newlines in strings
    clean_str = escape_newlines_in_strings(clean_str)
    
    return clean_str


def aggressive_json_repair(json_str: str) -> Dict[str, Any]:
    """
    Aggressively attempt to extract JSON from malformed text.
    Used as a last resort when other methods fail.
    
    Args:
        json_str: String that may contain malformed JSON
        
    Returns:
        A dictionary with best-effort extraction of key values
    """
    # Create a fallback empty result
    result = {}
    
    # Try to find decision pattern
    decision_match = re.search(r'"decision"\s*:\s*"([^"]*)"', json_str)
    if decision_match:
        result["decision"] = decision_match.group(1)
    
    # Try to find message pattern
    message_match = re.search(r'"message"\s*:\s*"([^"]*)"', json_str)
    if message_match:
        result["message"] = message_match.group(1)
    
    # Try to find answer pattern
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', json_str)
    if answer_match:
        result["answer"] = answer_match.group(1)
        
    # If nothing found, provide a default
    if not result:
        result = {
            "decision": "provide_final_answer",
            "message": "Could not parse response.",
            "answer": "Failed to parse JSON response."
        }
    
    return result 