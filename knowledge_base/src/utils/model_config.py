#!/usr/bin/env python3
import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from pydantic import SecretStr

from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(
        self,
        use_local_embeddings: bool = True,
        embedding_model: str = "nomic-embed-text",
    ):
        """
        Initialize model configuration.

        Args:
            use_local_embeddings: Whether to use local Ollama embeddings
            embedding_model: Ollama embedding model to use
        """
        self.use_local_embeddings = use_local_embeddings
        self.embedding_model = embedding_model

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.gemini_api_key: str = gemini_api_key

    def get_embeddings(self):
        if self.use_local_embeddings:
            return OllamaEmbeddings(
                model=self.embedding_model,
                num_thread=2,
            )
        else:
            # Fallback to Gemini
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            logger.info("Using Gemini API embeddings")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=SecretStr(self.gemini_api_key),
            )

    def get_gemini_llm(self, model: str = "gemini-1.5-flash"):
        """
        Get Gemini LLM for generation.

        Args:
            model: Gemini model to use

        Returns:
            Configured Gemini LLM
        """
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=SecretStr(self.gemini_api_key),
        )

    def get_local_llm(self, model: str = "llama3.2:3b"):
        """
        Get local Ollama LLM.

        Args:
            model: Ollama model to use

        Returns:
            Local Ollama LLM
        """
        from langchain_ollama import OllamaLLM

        logger.warning(f"Using local Ollama LLM: {model}")
        return OllamaLLM(
            model=model,
            num_thread=2,
        )

    def test_models(self):
        logger.info("Testing model configuration...")

        try:
            embeddings = self.get_embeddings()
            test_embedding = embeddings.embed_query("test query")
            logger.info(f"Embeddings working (dimension: {len(test_embedding)})")

            llm = self.get_gemini_llm()
            response = llm.invoke("Say 'Hello, models are working!'")
            logger.info(f"LLM working: {response.content[:50]}...")

            return True

        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False


def hybrid_model_config() -> ModelConfig:
    """
    Get Ollama embeddings and Gemini 1.5 flash for LLM.

    Returns:
        ModelConfig instance.
    """
    return ModelConfig()


if __name__ == "__main__":
    model_config = hybrid_model_config()
    success = model_config.test_models()

    if success:
        print("Hybrid model configuration is working!")
        print("Using local Ollama for embeddings")
        print("Using Gemini API for LLM generation")
    else:
        print("Model configuration test failed")
        print("Check that Ollama is running and Gemini API key is set")
