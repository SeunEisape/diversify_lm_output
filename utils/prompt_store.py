"""
Centralized prompt store for text generation scripts.
This module contains common prompts used across different generation scripts.
"""

# Prompt bank - collection of different prompts that can be used
PROMPT_BANK = {
    "creative_story": "Write a 500-word creative story:",
    
    "default": "The United States of America (USA), also known as the United States (U.S.) or America, "
                          "is a country primarily located in North America. It",
    
    "NLP_research": "Write a numbered list of 100 new research projects in natural language processing:"
                    "1. diversyfying the open source language model output "
                    "2. Finding correlation between the human brain and language models "
                    "3. ",
    
    "NLP_research_no_examples": "Write a numbered list of 100 new research projects in natural language processing:"
                                "1. ",
    
    "haiku": "Write a haiku:",
    
    # "default": "Write a 500-word creative story:",
    
    "poem": "Write a 250-word poem:"
}

def get_prompt(key, default_key="default"):
    """
    Get a prompt by key from the prompt bank.
    
    Args:
        key (str): The prompt key to retrieve
        default_key (str): Fallback key if the requested key is not found
        
    Returns:
        str: The prompt text
    """
    return PROMPT_BANK.get(key, PROMPT_BANK.get(default_key, ""))

def get_all_prompts():
    """
    Get all available prompts.
    
    Returns:
        dict: The complete prompt bank
    """
    return PROMPT_BANK 