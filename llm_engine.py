"""
LLM Engine - Model Factory for switching between Cloud (OpenAI) and Local (Ollama) backends.
Uses LangChain's ChatOpenAI with configurable endpoints.
"""

import os
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Factory class for creating LLM clients with different backends.

    Supports:
    - Cloud: OpenAI-compatible APIs (OpenAI, Volcano Engine/Ark, DeepSeek, etc.)
    - Local: Ollama or any OpenAI-compatible local server
    """

    def __init__(
        self,
        mode: Literal["cloud", "local"] = "cloud",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM client.

        Args:
            mode: "cloud" for OpenAI-compatible APIs, "local" for Ollama/local server
            model_name: Model name or endpoint ID (e.g., "gpt-4o", "ep-2025...", "deepseek-chat")
            base_url: Base URL for API (e.g., "https://ark.cn-beijing.volces.com/api/v3",
                     "https://api.deepseek.com", or "http://localhost:11434/v1")
            api_key: API key (for cloud providers or dummy key for local)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.mode = mode
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the appropriate LLM
        self.llm = self._create_llm(base_url, api_key)

    def _create_llm(self, base_url: Optional[str], api_key: Optional[str]) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance based on the mode.

        Args:
            base_url: Base URL for API
            api_key: API key

        Returns:
            Configured ChatOpenAI instance
        """
        if self.mode == "cloud":
            # Cloud mode - Use OpenAI-compatible API
            # Priority for base_url: method argument > env var > None (OpenAI official)
            url = base_url or os.getenv("OPENAI_BASE_URL")

            # Model name: use provided value or env var, but allow endpoint IDs
            if self.model_name:
                model = self.model_name
            else:
                env_model = os.getenv("OPENAI_MODEL")
                model = env_model if env_model else None  # Don't force default if user provided one

            key = api_key or os.getenv("OPENAI_API_KEY")

            if not key:
                raise ValueError(
                    "API key not found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            if not model:
                raise ValueError(
                    "Model name not found. Set OPENAI_MODEL environment variable "
                    "or pass model_name parameter."
                )

            logger.info(f"Initializing Cloud LLM: {model} at {url or 'OpenAI official'}")
            return ChatOpenAI(
                model=model,
                api_key=key,
                base_url=url,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.mode == "local":
            # Local mode - Use Ollama or compatible local server
            model = self.model_name or os.getenv("LOCAL_MODEL", "qwen2.5:14b")
            url = base_url or os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
            key = api_key or "ollama"  # Dummy key for local servers

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
            system_message: Optional system message

        Returns:
            LLM response as string
        """
        messages = []

        if system_message:
            messages.append(SystemMessage(content=system_message))

        messages.append(HumanMessage(content=message))

        response = self.llm.invoke(messages)
        return response.content

    def get_mode(self) -> str:
        """Get the current mode ('cloud' or 'local')."""
        return self.mode

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.llm.model_name


def create_llm_from_config(config: dict) -> LLMClient:
    """
    Create an LLMClient from a configuration dictionary.

    Args:
        config: Dictionary with keys:
            - mode: "cloud" or "local"
            - model_name: str (optional)
            - base_url: str (optional, for local)
            - api_key: str (optional)
            - temperature: float (optional)
            - max_tokens: int (optional)

    Returns:
        Configured LLMClient instance
    """
    return LLMClient(
        mode=config.get("mode", "cloud"),
        model_name=config.get("model_name"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens")
    )


# Default system prompt for investment assistant
DEFAULT_SYSTEM_PROMPT = """
You are OdinOracle, an AI-powered investment assistant. You help users track their portfolio,
analyze stocks, and make informed investment decisions.

You have access to tools for:
- Getting real-time stock prices and information
- Searching for recent stock news

When responding to users:
1. Be helpful, informative, and objective
2. Always clarify that you are an AI and not a financial advisor
3. Provide data-driven insights when available
4. Highlight risks and uncertainties
5. Avoid giving specific investment recommendations or advice

If you don't have sufficient information, be honest and suggest what data would be helpful.
"""


if __name__ == "__main__":
    # Test the LLM engine
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing LLM Engine...")
    print("=" * 60)

    # Test Cloud mode (requires OPENAI_API_KEY)
    try:
        print("\n1. Testing Cloud mode:")
        cloud_llm = LLMClient(mode="cloud", model_name="gpt-4o")
        response = cloud_llm.invoke("What is the current price of NVDA?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Cloud mode error: {e}")

    # Test Local mode (requires local server running)
    try:
        print("\n2. Testing Local mode:")
        local_llm = LLMClient(
            mode="local",
            model_name="qwen2.5:14b",
            base_url="http://localhost:11434/v1"
        )
        response = local_llm.invoke("Hello, can you help me with stocks?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Local mode error: {e}")
        print("Make sure Ollama or similar server is running at localhost:11434")
