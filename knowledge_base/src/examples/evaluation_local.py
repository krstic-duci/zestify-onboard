#!/usr/bin/env python3
"""
Quota-Friendly RAG Evaluation using Local Models

This script evaluates the RAG system using only local Ollama models to avoid API quotas.
Perfect for testing when you've hit Gemini limits.

Usage:
    uv run python evaluation_local.py
"""

import asyncio
import logging
from typing import List

from src.data_embedding import get_vector_store_stats, initialize_vector_store
from src.utils.model_config import get_optimized_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LocalRAGEvaluator:
    """
    RAG evaluator using only local models (no API quota limits).

    Uses local Ollama for both embeddings and LLM generation.
    """

    def __init__(self):
        """Initialize local RAG evaluator."""
        # Test questions about Zestify
        self.test_questions = [
            "What is Zestify and what does it do?",
            "What programming language is Zestify built with?",
            "How does Zestify handle authentication?",
            "What database does Zestify use?",
            "What are the main features of Zestify?",
        ]

        # Get optimized config
        self.config = get_optimized_config()

        # Local embeddings (fast, no quota)
        self.embeddings = self.config.get_embeddings()

        # Local LLM (slow but quota-free)
        self.local_llm = self.config.get_local_llm("llama3.2:1b")

        # Initialize vector store
        self.vector_store = initialize_vector_store()

    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant context for a query using local embeddings.

        Args:
            query: Question to retrieve context for
            k: Number of documents to retrieve

        Returns:
            List of relevant document texts
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving context for query '{query}': {e}")
            return []

    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate answer using local LLM and retrieved context.

        Args:
            query: Question to answer
            context: Retrieved context documents

        Returns:
            Generated answer
        """
        try:
            context_text = "\n\n".join(context[:2])  # Limit context for small model
            prompt = f"""Based on the following information, answer the question briefly and accurately.

Context:
{context_text}

Question: {query}

Answer:"""

            logger.info(f"Generating answer for: {query[:50]}...")
            response = self.local_llm.invoke(prompt)
            return str(response).strip()
        except Exception as e:
            logger.error(f"Error generating answer for query '{query}': {e}")
            return f"Error generating answer: {e}"

    def evaluate_rag_system(self):
        """
        Evaluate RAG system using only local models.

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting local RAG evaluation...")

        results = []

        for i, query in enumerate(self.test_questions, 1):
            logger.info(
                f"Processing question {i}/{len(self.test_questions)}: {query[:30]}..."
            )

            try:
                # Retrieve context
                retrieved_contexts = self.retrieve_context(query)

                if not retrieved_contexts:
                    logger.warning(f"No context retrieved for: {query}")
                    continue

                # Generate response
                response = self.generate_answer(query, retrieved_contexts)

                # Store result
                result = {
                    "question": query,
                    "retrieved_contexts": len(retrieved_contexts),
                    "response": response,
                    "response_length": len(response),
                }

                results.append(result)

                # Show progress
                print(f"\n📝 Question {i}: {query}")
                print(f"🔍 Retrieved {len(retrieved_contexts)} relevant documents")
                print(
                    f"🤖 Answer: {response[:100]}{'...' if len(response) > 100 else ''}"
                )

            except Exception as e:
                logger.error(f"Error processing question '{query}': {e}")
                continue

        return results

    def print_evaluation_summary(self, results):
        """
        Print evaluation results summary.

        Args:
            results: List of evaluation results
        """
        print("\n" + "=" * 60)
        print("LOCAL RAG EVALUATION RESULTS")
        print("=" * 60)

        if not results:
            print("❌ No results generated")
            return

        print(f"📊 Evaluated {len(results)} questions successfully")

        # Calculate metrics
        avg_contexts = sum(r["retrieved_contexts"] for r in results) / len(results)
        avg_response_length = sum(r["response_length"] for r in results) / len(results)

        print(f"📈 Average contexts retrieved: {avg_contexts:.1f}")
        print(f"📏 Average response length: {avg_response_length:.0f} characters")

        # Show sample results
        print("\n📋 Sample Q&A:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Q: {result['question']}")
            print(
                f"   A: {result['response'][:150]}{'...' if len(result['response']) > 150 else ''}"
            )

        print("\n" + "=" * 60)
        print("✅ Local evaluation completed successfully!")
        print("💡 Benefits of local evaluation:")
        print("  • No API quota consumption")
        print("  • Unlimited testing capacity")
        print("  • Works offline")
        print("  • Good for development/testing")


async def main():
    """Main function to run local RAG evaluation."""
    logger.info("Starting quota-free RAG evaluation...")

    # Check vector store status
    try:
        vector_store = initialize_vector_store()
        stats = get_vector_store_stats(vector_store)
        logger.info(f"Vector store ready with {stats['total_documents']} documents")

        if stats["total_documents"] == 0:
            print("\n⚠️  No data in vector store!")
            print("Run the ingestion pipeline first: uv run python main.py")
            return

    except Exception as e:
        logger.error(f"Vector store not ready: {e}")
        return

    # Initialize evaluator
    print("🤖 Using Local Models Only (No API Quota)")
    print("=" * 45)

    evaluator = LocalRAGEvaluator()

    # Run evaluation
    try:
        results = evaluator.evaluate_rag_system()
        evaluator.print_evaluation_summary(results)

        logger.info("Local evaluation completed!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
