#!/usr/bin/env python3
import asyncio
import logging
from typing import List

from ragas import EvaluationDataset, evaluate  # type: ignore
from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
from ragas.llms import LangchainLLMWrapper  # type: ignore
from ragas.metrics import (  # type: ignore
    AnswerRelevancy,
    Faithfulness,
    LLMContextRecall,
)

from src.data_embedding import get_vector_store_stats, initialize_vector_store
from src.utils.logging_config import setup_logging
from src.utils.model_config import get_optimized_config

setup_logging()
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG system performance using Ragas framework.

    Attributes:
        test_questions: List of test questions for evaluation
        expected_answers: List of expected answers for comparison
        vector_store: ChromaDB vector store for retrieval
        llm: Language model for generation
        evaluator_llm: LLM for evaluation
        embeddings: Embedding model
    """

    def __init__(self):
        """Initialize RAG evaluator with test data and models."""
        self.test_questions = [
            "What is Zestify and what problem does it solve?",
            "What programming language and framework is Zestify built with?",
            "How does Zestify categorize ingredients from recipes?",
            "What AI model does Zestify use for ingredient processing?",
            "What security features are implemented in Zestify?",
            "How does the rate limiting work in Zestify?",
            "What is the weekly meal planner feature in Zestify?",
            "How does authentication work in the Zestify application?",
            "What database does Zestify use and how is it configured?",
            "What are the main API endpoints available in Zestify?",
        ]

        self.expected_answers = [
            "Zestify is a recipe ingredient aggregator that helps organize and categorize ingredients from multiple recipes. It solves the problem of managing ingredients across multiple recipes by extracting, translating them to Swedish, and organizing them by category.",
            "Zestify is built with Python using the FastAPI framework. It uses Python 3.13+ and includes modern async/await patterns.",
            "Zestify automatically groups ingredients into categories like Meat/Fish, Vegetables/Fruits, Dairy, Grains, Spices/Herbs, and Other using Google's Gemini AI for smart categorization.",
            "Zestify uses Google Gemini 1.5 Flash AI model for ingredient processing, extraction, and categorization.",
            "Zestify implements bcrypt password hashing, signed tokens, rate limiting, HTML sanitization with Bleach, secure cookies (HttpOnly, Secure, SameSite=Strict), and comprehensive error handling.",
            "Zestify has custom FastAPI rate limiting: 5 requests per minute for login endpoint and 10 requests per minute for general endpoints, implemented per IP address.",
            "The weekly meal planner allows users to drag-and-drop meals for each day and meal type, with endpoints for swapping and moving meals to organize weekly meal planning.",
            "Authentication uses session-based auth with signed tokens, bcrypt password hashing, and protected routes that require login to access the main functionality.",
            "Zestify uses Supabase (Postgres) as its database and includes a database connection module in the db/ directory.",
            "Main API endpoints include GET / (ingredient input), POST /login (authentication), POST /ingredients (recipe processing), GET /weekly (meal planner), and POST /swap-meals and POST /move-meal for meal management.",
        ]

        # Initialize models using hybrid configuration
        config = get_optimized_config()
        self.llm = config.get_llm()
        self.embeddings = config.get_embeddings()

        # Wrap for Ragas
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        # Initialize vector store
        self.vector_store = initialize_vector_store()

    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant context for a query.

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
        Generate answer using retrieved context.

        Args:
            query: Question to answer
            context: Retrieved context documents

        Returns:
            Generated answer
        """
        try:
            context_text = "\n\n".join(context)
            prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context_text}

Question: {query}

Answer:"""

            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, str):
                return content.strip()
            else:
                return str(content).strip()
        except Exception as e:
            logger.error(f"Error generating answer for query '{query}': {e}")
            return "Error generating answer"

    def create_evaluation_dataset(self) -> EvaluationDataset:
        """
        Create evaluation dataset by running queries through RAG system.

        Returns:
            EvaluationDataset for ragas evaluation
        """
        dataset = []

        for query, expected in zip(
            self.test_questions, self.expected_answers, strict=True
        ):
            logger.info(f"Processing query: {query[:50]}...")

            # Retrieve context
            retrieved_contexts = self.retrieve_context(query)

            # Generate response
            response = self.generate_answer(query, retrieved_contexts)

            # Create evaluation sample
            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": retrieved_contexts,
                    "response": response,
                    "reference": expected,
                }
            )

        return EvaluationDataset.from_list(dataset)

    async def evaluate_rag_system(self):
        """
        Evaluate RAG system using Ragas metrics.

        Returns:
            Evaluation results from ragas
        """
        logger.info("Creating evaluation dataset...")
        evaluation_dataset = self.create_evaluation_dataset()

        logger.info("Setting up evaluation metrics...")
        metrics = [
            LLMContextRecall(llm=self.evaluator_llm),
            Faithfulness(llm=self.evaluator_llm),
            AnswerRelevancy(
                llm=self.evaluator_llm, embeddings=self.evaluator_embeddings
            ),
        ]

        logger.info("Running evaluation...")
        result = evaluate(
            dataset=evaluation_dataset, metrics=metrics, llm=self.evaluator_llm
        )

        return result

    def print_evaluation_summary(self, result):
        """
        Print evaluation results summary.

        Args:
            result: Evaluation results from ragas
        """
        print("\n" + "=" * 60)
        print("RAG SYSTEM EVALUATION RESULTS")
        print("=" * 60)

        # Overall scores
        print("\nOverall Scores:")
        # EvaluationResult has attributes, not dictionary items
        if hasattr(result, "to_pandas"):
            try:
                # Get metrics as dictionary by accessing result attributes
                metrics_dict = {}
                for attr_name in dir(result):
                    if not attr_name.startswith("_") and not callable(
                        getattr(result, attr_name)
                    ):
                        attr_value = getattr(result, attr_name)
                        if isinstance(attr_value, (int, float)):
                            metrics_dict[attr_name] = attr_value

                for metric, score in metrics_dict.items():
                    print(f"  {metric}: {score:.4f}")

            except Exception as e:
                logger.warning(f"Could not extract metric scores: {e}")
                print("  Unable to extract individual metric scores")

        # Convert to DataFrame for detailed analysis
        try:
            df = result.to_pandas()
            print(f"\nDetailed Results ({len(df)} samples):")
            print(df.to_string(index=False))

            # Summary statistics
            print("\nSummary Statistics:")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                mean_score = df[col].mean()
                min_score = df[col].min()
                max_score = df[col].max()
                print(f"  {col}:")
                print(f"    Mean: {mean_score:.4f}")
                print(f"    Min:  {min_score:.4f}")
                print(f"    Max:  {max_score:.4f}")
        except Exception as e:
            logger.warning(f"Could not generate detailed results: {e}")

        print("\n" + "=" * 60)


async def main():
    """Main function to run RAG evaluation."""
    logger.info("Starting RAG system evaluation...")

    # Check vector store status
    try:
        # Initialize vector store to check if it exists
        vector_store = initialize_vector_store()
        stats = get_vector_store_stats(vector_store)
        logger.info(f"Vector store ready with {stats['total_documents']} documents")
    except Exception as e:
        logger.error(f"Vector store not ready: {e}")
        logger.error("Please run the ingestion pipeline first: uv run python main.py")
        return

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Run evaluation
    try:
        result = await evaluator.evaluate_rag_system()
        evaluator.print_evaluation_summary(result)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
