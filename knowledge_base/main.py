import logging
from pathlib import Path

from src.data_split import split_documents
from src.git_loader import load_git_repository

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function to load, clean, and process documents from a Git repository,
    and save cleaned examples for inspection.
    """
    logging.info("Loading documents from Git repository...")
    documents = load_git_repository()
    logging.info(f"Loaded {len(documents)} documents.")

    if not documents:
        logging.warning("No documents were loaded. Exiting.")
        return

    # Clean and split documents
    all_chunks = split_documents(documents)
    logging.info(f"Total chunks generated: {len(all_chunks)}")

    # --- Save cleaned and split examples for inspection ---
    output_dir = Path("cleaned_examples")
    output_dir.mkdir(exist_ok=True)
    logging.info(f"Saving cleaned and split examples to: {output_dir.resolve()}")

    # Clear previous cleaned examples
    for f in output_dir.glob("clean_*.txt"):
        f.unlink()

    for i, chunk in enumerate(all_chunks):
        # Create a unique filename for each chunk
        original_filename = Path(chunk.metadata.get("source", "unknown")).name
        chunk_filename = f"chunk_{i}_{original_filename}.txt"
        output_path = output_dir / chunk_filename

        try:
            output_path.write_text(chunk.page_content, encoding="utf-8")
        except Exception as e:
            logging.error(f"Error writing chunk {chunk_filename}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
