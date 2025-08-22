import logging

from src.data_embedding import (
    add_documents_to_store,
    get_vector_store_stats,
    initialize_vector_store,
)
from src.data_load import load_git_repository
from src.data_split import split_documents
from src.utils.logging_config import setup_logging
from src.utils.save_cleaned_examples import save_cleaned_examples

setup_logging()
logger = logging.getLogger(__name__)

SAVE_EXAMPLES = False  # Set to False to skip saving example files


def ingest():
    """
    Main function to load, clean, process documents from a Git repository,
    embed them, and store in ChromaDB vector store.
    """
    # Step 1: Load documents from Zestify Git repository
    documents = load_git_repository()

    if not documents:
        logger.warning("No documents were loaded. Exiting.")
        return

    # Step 2: Clean and split documents
    all_chunks = split_documents(documents)

    if not all_chunks:
        logger.warning("No chunks were generated. Exiting.")
        return

    # Step 3: Initialize Chroma vector store
    try:
        vector_store = initialize_vector_store()
    except Exception as e:
        logger.error("Failed to initialize Chroma vector store: %s", e)
        return

    # Step 4: Get current Chroma vector store statistics
    stats_before = get_vector_store_stats(vector_store)
    logger.info("Chroma vector store stats BEFORE ingestion: %s", stats_before)

    # Step 5: Add documents to vector store
    logger.info("Adding %d chunks to Chroma vector store...", len(all_chunks))
    added_count = add_documents_to_store(vector_store, all_chunks)

    if added_count > 0:
        logger.info("Successfully added %d chunks to Chroma vector store.", added_count)
    else:
        logger.error("Failed to add any documents to Chroma vector store.")
        return

    # Step 6: Get updated vector store statistics
    stats_after = get_vector_store_stats(vector_store)
    logger.info("Chroma vector store stats AFTER ingestion: %s", stats_after)

    if SAVE_EXAMPLES:
        save_cleaned_examples(all_chunks)

    logger.info("Ingestion pipeline completed successfully!")


if __name__ == "__main__":
    try:
        ingest()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        exit(0)
    except Exception as e:
        logger.error("Error occurred during ingestion: %s", e)
