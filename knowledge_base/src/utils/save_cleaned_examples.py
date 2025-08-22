import logging
from pathlib import Path

from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MAX_EXAMPLES = 500


def save_cleaned_examples(chunks: list) -> None:
    """
    Save cleaned and split examples for inspection.

    Args:
        chunks: List of document chunks to save as examples.
    """
    output_dir = Path("cleaned_examples")
    output_dir.mkdir(exist_ok=True)
    logger.info("Saving cleaned examples to: %s", output_dir.resolve())

    for existing_file in output_dir.glob("chunk_*.txt"):
        existing_file.unlink()

    sample_chunks = chunks[:MAX_EXAMPLES] if len(chunks) > MAX_EXAMPLES else chunks

    for i, chunk in enumerate(sample_chunks):
        original_filename = Path(chunk.metadata.get("source", "unknown")).name
        element_type = chunk.metadata.get("element_type", "chunk")
        element_name = chunk.metadata.get("element_name", "")

        if element_name:
            chunk_filename = (
                f"chunk_{i:03d}_{element_type}_{element_name}_{original_filename}.txt"
            )
        else:
            chunk_filename = f"chunk_{i:03d}_{element_type}_{original_filename}.txt"

        chunk_filename = "".join(c for c in chunk_filename if c.isalnum() or c in "._-")
        output_path = output_dir / chunk_filename

        try:
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
