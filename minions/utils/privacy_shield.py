"""
Module for applying privacy protection to LLM outputs.
"""

from typing import Optional, Dict, Any
from minions.clients.base import BaseClient

def apply_privacy_shield(
    response: str, 
    client: BaseClient,
    pii_json: Optional[str] = None
) -> str:
    """
    Apply privacy shield to remove sensitive information from responses.
    
    Args:
        response: The original response text to shield
        client: The LLM client to use for rewriting
        pii_json: Optional JSON string with PII information to avoid
        
    Returns:
        Privacy-protected response with sensitive information removed
    """
    # Create a prompt for privacy-focused rewriting
    privacy_prompt = f"""
You are a privacy protection assistant. Your task is to rewrite the following text
to remove or redact any potentially sensitive or personal information while preserving
the meaning and informativeness of the content.

If you encounter names, addresses, phone numbers, emails, financial information, or other
personally identifiable information (PII), replace them with generic placeholders.

Original text:
{response}

Additional PII information to avoid:
{pii_json or "None specified"}

Rewrite the text to protect privacy while maintaining the same information content:
"""
    
    # Send to the model for rewriting
    try:
        shielded_responses, _, _ = client.chat([{"role": "user", "content": privacy_prompt}])
        return shielded_responses[0]
    except Exception as e:
        # If there's an error, return the original with a warning
        print(f"Error applying privacy shield: {e}")
        return f"[PRIVACY SHIELD ERROR - using original response] {response}" 