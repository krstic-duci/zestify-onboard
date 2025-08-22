#!/usr/bin/env python3
import logging
from typing import Any, Dict, List

from langchain_chroma import Chroma

from src.data_embedding import initialize_vector_store
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def get_collection_sample(vector_store: Chroma, limit: int = 2) -> Dict[str, List[Any]]:
    """
    Retrieve a sample of records from the Chroma collection.

    Args:
        vector_store: The Chroma vector store instance.
        limit: The number of records to retrieve.

    Returns:
        A dictionary containing a sample of the collection data.
    """
    try:
        collection = vector_store._collection
        count = collection.count()

        if count == 0:
            logger.info("The collection is empty.")
            return {}

        sample = collection.peek(limit=min(limit, count))
        return sample  # type: ignore

    except Exception as e:
        logger.error(f"Error retrieving collection sample: {e}", exc_info=True)
        return {}


def print_sample_records(sample: Dict[str, List[Any]]) -> None:
    """
    Print the sample records in a readable format.

    Args:
        sample: A dictionary containing the collection data.
    """
    if not sample:
        logger.warning("No sample data to print.")
        return

    logger.info("Visualizing 1-2 rows from the ChromaDB vector store:")

    ids = sample.get("ids", [])
    metadatas = sample.get("metadatas", [])
    documents = sample.get("documents", [])
    embeddings = sample.get("embeddings")

    for i, doc_id in enumerate(ids):
        print(f"--- Record {i + 1} ---")
        print(f"  ID: {doc_id}")

        if metadatas and len(metadatas) > i:
            metadata = metadatas[i]
            if metadata:
                print("  Metadata:")
                for key, value in metadata.items():
                    print(f"    {key}: {value}")

        if documents and len(documents) > i:
            document = documents[i]
            print(f"  Chunk Text:\n'''{document}'''")

        if embeddings is not None and len(embeddings) > i:
            embedding = embeddings[i]
            if embedding is not None:
                print(f"  Embedding (first 10 elements): {embedding[:10]}")
        print("\n")


def main():
    """
    Main function to initialize the vector store and visualize its content.
    """
    logger.info("Initializing vector store...")
    try:
        vector_store = initialize_vector_store()
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        return

    logger.info("Retrieving collection sample...")
    sample = get_collection_sample(vector_store, limit=2)

    if sample:
        print_sample_records(sample)
    else:
        logger.warning("Could not retrieve any records from the vector store.")


if __name__ == "__main__":
    main()
