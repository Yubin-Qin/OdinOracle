"""
LLM Engine - Model Factory and Aggressive Hedge Fund AI Persona.
Supports OpenAI-compatible APIs with bilingual output (English/Chinese).
Refactored to use centralized configuration and externalized prompts.
"""

from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from config import get_settings
from prompts import get_system_prompt, DEFAULT_SYSTEM_PROMPT_EN, DEFAULT_SYSTEM_PROMPT_ZH

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Factory class for creating LLM clients with different backends.
    Supports bilingual output (English/Chinese).
    Uses centralized configuration from config.py and externalized prompts.
    """

    def __init__(
        self,
        mode: Literal["cloud", "local"] = "cloud",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        language: str = "en"
    ):
        """
        Initialize the LLM client.

        Args:
            mode: "cloud" for OpenAI-compatible APIs, "local" for Ollama/local server
            model_name: Model name or endpoint ID (e.g., "gpt-4o", "ep-2025...", "deepseek-chat")
            base_url: Base URL for API
            api_key: API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            language: Output language ("en" or "zh")
        """
        self.mode = mode
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.language = language

        # Initialize the appropriate LLM
        self.llm = self._create_llm(base_url, api_key)

    def _create_llm(self, base_url: Optional[str], api_key: Optional[str]) -> ChatOpenAI:
        """Create a ChatOpenAI instance based on the mode."""
        settings = get_settings()

        if self.mode == "cloud":
            url = base_url or settings.openai_base_url

            if self.model_name:
                model = self.model_name
            else:
                model = settings.openai_model

            key = api_key or settings.openai_api_key

            if not key:
                raise ValueError("API key not found. Set OPENAI_API_KEY environment variable.")
            if not model:
                raise ValueError("Model name not found. Set OPENAI_MODEL environment variable.")

            logger.info(f"Initializing Cloud LLM: {model} at {url or 'OpenAI official'}")
            return ChatOpenAI(
                model=model,
                api_key=key,
                base_url=url,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.mode == "local":
            model = self.model_name or settings.local_model
            url = base_url or settings.local_llm_url
            key = api_key or "ollama"

            logger.info(f"Initializing Local LLM: {model} at {url}")
            return ChatOpenAI(
                model=model,
                base_url=url,
                api_key=key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'cloud' or 'local'.")

    def get_llm(self) -> ChatOpenAI:
        """Get the LangChain ChatOpenAI instance."""
        return self.llm

    def invoke(self, message: str, system_message: Optional[str] = None) -> str:
        """
        Invoke the LLM with a message.

        Args:
            message: User message
            system_message: Optional custom system message (overrides default)

        Returns:
            LLM response as string
        """
        messages = []

        # Use appropriate system prompt based on language
        if system_message:
            messages.append(SystemMessage(content=system_message))
        else:
            # Use cached prompts for better performance
            prompt = DEFAULT_SYSTEM_PROMPT_ZH if self.language == "zh" else DEFAULT_SYSTEM_PROMPT_EN
            messages.append(SystemMessage(content=prompt))

        messages.append(HumanMessage(content=message))

        response = self.llm.invoke(messages)
        return response.content

    def get_mode(self) -> str:
        """Get the current mode ('cloud' or 'local')."""
        return self.mode

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.llm.model_name

    def set_language(self, language: str):
        """Set the output language ('en' or 'zh')."""
        self.language = language


def get_system_prompt(language: str = "en") -> str:
    """Get the appropriate system prompt based on language."""
    return DEFAULT_SYSTEM_PROMPT_ZH if language == "zh" else DEFAULT_SYSTEM_PROMPT_EN


def create_llm_from_config(config: dict) -> LLMClient:
    """Create an LLMClient from a configuration dictionary."""
    return LLMClient(
        mode=config.get("mode", "cloud"),
        model_name=config.get("model_name"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens"),
        language=config.get("language", "en")
    )


if __name__ == "__main__":
    # Test the LLM engine
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing LLM Engine...")
    print("=" * 60)

    # Test Cloud mode (requires OPENAI_API_KEY)
    try:
        print("\n1. Testing Cloud mode (English):")
        cloud_llm = LLMClient(mode="cloud", model_name="gpt-4o", language="en")
        response = cloud_llm.invoke("Analyze NVDA's current setup. Give me a signal and confidence score.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Cloud mode error: {e}")

    # Test Local mode
    try:
        print("\n2. Testing Local mode (Chinese):")
        local_llm = LLMClient(
            mode="local",
            model_name="qwen2.5:14b",
            base_url="http://localhost:11434/v1",
            language="zh"
        )
        response = local_llm.invoke("分析NVDA当前的走势，给我信号和信心评分。")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Local mode error: {e}")
