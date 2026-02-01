"""
Prompts package for OdinOracle.
Contains system prompts for LLM personalities.
"""

import os
from typing import Dict

# Cache for loaded prompts
_prompt_cache: Dict[str, str] = {}


def load_prompt(filename: str) -> str:
    """
    Load a prompt from a text file.

    Args:
        filename: Name of the prompt file (e.g., 'system_prompt_en.txt')

    Returns:
        Prompt content as string
    """
    if filename in _prompt_cache:
        return _prompt_cache[filename]

    # Get the directory where this file is located
    prompt_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(prompt_dir, filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            _prompt_cache[filename] = content
            return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading prompt file {filename}: {e}")


def get_system_prompt(language: str = "en") -> str:
    """
    Get the system prompt for the specified language.

    Args:
        language: 'en' for English, 'zh' for Chinese

    Returns:
        System prompt content
    """
    filename = f"system_prompt_{language}.txt"
    return load_prompt(filename)


def clear_prompt_cache():
    """Clear the prompt cache. Useful for reloading prompts during development."""
    _prompt_cache.clear()


# Pre-load common prompts for better performance
DEFAULT_SYSTEM_PROMPT_EN = load_prompt("system_prompt_en.txt")
DEFAULT_SYSTEM_PROMPT_ZH = load_prompt("system_prompt_zh.txt")
