import logging
from pathlib import Path

from src.data_embedding import (
    add_documents_to_store,
    get_vector_store_stats,
    initialize_vector_store,
)
from src.data_load import load_git_repository
from src.data_split import split_documents

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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


def save_cleaned_examples(chunks: list) -> None:
    """
    Save cleaned and split examples for inspection.

    Args:
        chunks: List of document chunks to save as examples.
    """
    output_dir = Path("cleaned_examples")
    output_dir.mkdir(exist_ok=True)
    logger.info("Saving cleaned examples to: %s", output_dir.resolve())

    # Clear previous cleaned examples
    for existing_file in output_dir.glob("chunk_*.txt"):
        existing_file.unlink()

    # Save a sample of chunks (limit to prevent too many files)
    max_examples = 50
    sample_chunks = chunks[:max_examples] if len(chunks) > max_examples else chunks

    for i, chunk in enumerate(sample_chunks):
        # Create a unique filename for each chunk
        original_filename = Path(chunk.metadata.get("source", "unknown")).name
        element_type = chunk.metadata.get("element_type", "chunk")
        element_name = chunk.metadata.get("element_name", "")

        # Create descriptive filename
        if element_name:
            chunk_filename = (
                f"chunk_{i:03d}_{element_type}_{element_name}_{original_filename}.txt"
            )
        else:
            chunk_filename = f"chunk_{i:03d}_{element_type}_{original_filename}.txt"

        # Clean filename of invalid characters
        chunk_filename = "".join(c for c in chunk_filename if c.isalnum() or c in "._-")
        output_path = output_dir / chunk_filename

        try:
            # Write chunk content with metadata header
            content = "=== CHUNK METADATA ===\n"
            for key, value in chunk.metadata.items():
                content += f"{key}: {value}\n"
            content += f"\n=== CHUNK CONTENT ===\n{chunk.page_content}\n"

            output_path.write_text(content, encoding="utf-8")

        except Exception as e:
            logger.error("Error writing chunk %s: %s", chunk_filename, e, exc_info=True)

    logger.info(
        "Saved %d example chunks (out of %d total)", len(sample_chunks), len(chunks)
    )


if __name__ == "__main__":
    main()
