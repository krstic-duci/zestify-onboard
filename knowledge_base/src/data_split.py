import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.utils.clean_document import clean_document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logger = logging.getLogger(__name__)


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Cleans and splits documents into smaller, semantically meaningful chunks.

    Args:
        documents: List of Document objects to process.

    Returns:
        List of processed and split Document chunks with enriched metadata.
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    # For HTML/Jinja, Markdown, and other text files
    general_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []

    for doc in documents:
        source_path = doc.metadata.get("source", "")
        file_suffix = Path(source_path).suffix

        try:
            # Clean document (may return multiple documents for Python files)
            cleaned_docs = clean_document(doc)

            for cleaned_doc in cleaned_docs:
                # Skip splitting for Python elements that are already well-bounded
                if cleaned_doc.metadata.get("element_type") in ["function", "class"]:
                    # Add file type metadata and keep as single chunk
                    cleaned_doc.metadata.update(
                        {
                            "file_type": file_suffix.lstrip("."),
                            "char_length": len(cleaned_doc.page_content),
                        }
                    )
                    all_chunks.append(cleaned_doc)
                    continue

                # Determine which splitter to use for other content
                if file_suffix == ".py":
                    splitter = python_splitter
                elif file_suffix == ".js":
                    splitter = js_splitter
                else:  # Includes .jinja, .html, .md, and other text files
                    splitter = general_splitter

                chunks = splitter.split_documents([cleaned_doc])

                # Enrich each chunk with metadata
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "file_type": file_suffix.lstrip("."),
                            "char_length": len(chunk.page_content),
                        }
                    )
                    # Preserve existing metadata from cleaning
                    if "element_type" not in chunk.metadata:
                        chunk.metadata["element_type"] = "chunk"

                all_chunks.extend(chunks)

        except Exception as e:
            logger.error(
                "Error cleaning or splitting file %s: %s", source_path, e, exc_info=True
            )

    logger.info(
        "Processed %d documents into %d chunks", len(documents), len(all_chunks)
    )
    return all_chunks
