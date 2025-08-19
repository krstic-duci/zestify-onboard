import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.data_clean import clean_document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Cleans and splits documents into smaller, semantically meaningful chunks.
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
            cleaned_doc = clean_document(doc)

            # Determine which splitter to use
            if file_suffix == ".py":
                splitter = python_splitter
            elif file_suffix == ".js":
                splitter = js_splitter
            else:  # Includes .jinja, .html, .md, and other text files
                splitter = general_splitter

            chunks = splitter.split_documents([cleaned_doc])
            all_chunks.extend(chunks)

        except Exception as e:
            logging.error(
                f"Error cleaning or splitting file {source_path}: {e}", exc_info=True
            )

    return all_chunks
