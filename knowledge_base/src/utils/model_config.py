#!/usr/bin/env python3
"""
Model configuration for hybrid Ollama + Gemini setup.

This module provides optimized model selection for different hardware configurations.
For low-end Macs, we use local Ollama for embeddings (no API limits, faster)
and Gemini API for LLM generation (better quality, handles quota).
"""

import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class ModelConfig:
    """
    Hybrid model configuration optimized for different hardware setups.

    Strategy:
    - Embeddings: Local Ollama (no API quota, good quality, runs well on 16GB RAM)
    - LLM: Gemini API (better quality than small local models, handles quota limits)
    """

    def __init__(
        self,
        use_local_embeddings: bool = True,
        embedding_model: str = "nomic-embed-text",
    ):
        """
        Initialize model configuration.

        Args:
            use_local_embeddings: Whether to use local Ollama embeddings (recommended)
            embedding_model: Ollama embedding model to use
        """
        self.use_local_embeddings = use_local_embeddings
        self.embedding_model = embedding_model

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.gemini_api_key: str = gemini_api_key

    def get_embeddings(self):
        """

        Returns:
            Configured embeddings model
        """
        if self.use_local_embeddings:
            logger.info(f"Using local Ollama embeddings: {self.embedding_model}")
            return OllamaEmbeddings(
                model=self.embedding_model,
                # Optimize for i5
                num_thread=2,
            )
        else:
            # Fallback to Gemini (uses API quota)
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            logger.info("Using Gemini API embeddings")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=SecretStr(self.gemini_api_key),
            )

    def get_llm(self, model: str = "gemini-1.5-flash"):
        """
        Get LLM for generation (Gemini API recommended for quality).

        Args:
            model: Gemini model to use

        Returns:
            Configured LLM
        """
        logger.info(f"Using Gemini API LLM: {model}")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=SecretStr(self.gemini_api_key),
        )

    def get_local_llm(self, model: str = "llama3.2:3b"):
        """
        Get local Ollama LLM (for testing only - will be slow on your hardware).

        Args:
            model: Ollama model to use

        Returns:
            Local Ollama LLM
        """
        from langchain_ollama import OllamaLLM

        logger.warning(
            f"Using local Ollama LLM: {model} (will be slow on dual-core i5)"
        )
        return OllamaLLM(
            model=model,
            num_thread=2,  # Optimize for your dual-core
        )

    def test_models(self):
        """Test that both embedding and LLM models are working."""
        logger.info("Testing model configuration...")

        try:
            # Test embeddings
            embeddings = self.get_embeddings()
            test_embedding = embeddings.embed_query("test query")
            logger.info(f"✅ Embeddings working (dimension: {len(test_embedding)})")

            # Test LLM
            llm = self.get_llm()
            response = llm.invoke("Say 'Hello, models are working!'")
            logger.info(f"✅ LLM working: {response.content[:50]}...")

            return True

        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            return False


def get_optimized_config() -> ModelConfig:
    """
    Get optimized model configuration for current hardware.

    Returns:
        ModelConfig instance optimized for the system
    """
    # For dual-core i5 with 16GB RAM, prefer local embeddings + remote LLM
    return ModelConfig(
        use_local_embeddings=True,
        embedding_model="nomic-embed-text",  # Start with lighter model
    )


if __name__ == "__main__":
    # Test the configuration
    logging.basicConfig(level=logging.INFO)

    config = get_optimized_config()
    success = config.test_models()

    if success:
        print("\n🎉 Hybrid model configuration is working!")
        print("💡 Using local Ollama for embeddings (no API quota)")
        print("🌐 Using Gemini API for LLM generation (better quality)")
    else:
        print("\n❌ Model configuration test failed")
        print("Check that Ollama is running and Gemini API key is set")
