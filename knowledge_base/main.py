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


def main():
    """
    Main function to load, clean, process documents from a Git repository,
    embed them, and store in ChromaDB vector store.
    """
    logger.info("Starting knowledge base ingestion pipeline...")

    # Step 1: Load documents from Git repository
    logger.info("Loading documents from Git repository...")
    documents = load_git_repository()
    logger.info("Loaded %d documents.", len(documents))

    if not documents:
        logger.warning("No documents were loaded. Exiting.")
        return

    # Step 2: Clean and split documents
    logger.info("Cleaning and splitting documents...")
    all_chunks = split_documents(documents)
    logger.info(
        "Generated %d chunks from %d documents.", len(all_chunks), len(documents)
    )

    if not all_chunks:
        logger.warning("No chunks were generated. Exiting.")
        return

    # Step 3: Initialize vector store
    logger.info("Initializing vector store...")
    try:
        vector_store = initialize_vector_store()
    except Exception as e:
        logger.error("Failed to initialize vector store: %s", e)
        return

    # Step 4: Get current vector store statistics
    logger.info("Getting current vector store statistics...")
    stats_before = get_vector_store_stats(vector_store)
    logger.info("Vector store stats before ingestion: %s", stats_before)

    # Step 5: Add documents to vector store
    logger.info("Adding %d chunks to vector store...", len(all_chunks))
    added_count = add_documents_to_store(vector_store, all_chunks)

    if added_count > 0:
        logger.info("Successfully added %d chunks to vector store.", added_count)
    else:
        logger.error("Failed to add any documents to vector store.")
        return

    # Step 6: Get updated vector store statistics
    logger.info("Getting updated vector store statistics...")
    stats_after = get_vector_store_stats(vector_store)
    logger.info("Vector store stats after ingestion: %s", stats_after)

    # Optional: Save cleaned examples for inspection
    if SAVE_EXAMPLES:
        save_cleaned_examples(all_chunks)

    logger.info("Knowledge base ingestion pipeline completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        exit(0)
    except Exception as e:
        logger.error("Error occurred during main execution: %s", e)
