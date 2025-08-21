import logging
from pathlib import Path

from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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
