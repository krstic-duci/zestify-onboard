import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PERSIST_DIRECTORY = "./chroma_langchain_db"
COLLECTION_NAME = "zestify_knowledge_base"

logger = logging.getLogger(__name__)


def create_document_id(doc: Document) -> str:
    """
    Create a unique, deterministic ID for a document based on its content and metadata.

    Args:
        doc: Document object to create ID for.

    Returns:
        Unique document ID as a hex string.
    """
    # Create a hash based on source path, content, and key metadata
    source_path = doc.metadata.get("source", "")
    element_type = doc.metadata.get("element_type", "")
    element_name = doc.metadata.get("element_name", "")
    start_line = str(doc.metadata.get("start_line", ""))

    id_string = (
        f"{source_path}:{element_type}:{element_name}:{start_line}:{doc.page_content}"
    )

    doc_id = hashlib.sha256(id_string.encode("utf-8")).hexdigest()

    return doc_id


def initialize_vector_store() -> Chroma:
    """
    Initialize the ChromaDB vector store with optimized hybrid embeddings.

    Returns:
        Configured Chroma vector store instance.
    """
    try:
        from .utils.model_config import get_optimized_config

        # Get optimized embedding model (local Ollama preferred)
        config = get_optimized_config()
        embeddings = config.get_embeddings()

        # Create persist directory if it doesn't exist
        persist_dir = Path(CHROMA_PERSIST_DIRECTORY)
        persist_dir.mkdir(exist_ok=True)

        # Initialize Chroma vector store
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )

        logger.info(
            "Initialized vector store with collection '%s' at %s",
            COLLECTION_NAME,
            persist_dir.resolve(),
        )

        return vector_store

    except Exception as e:
        logger.error("Failed to initialize vector store: %s", e, exc_info=True)
        raise


def add_documents_to_store(
    vector_store: Chroma, documents: List[Document], batch_size: int = 100
) -> int:
    """
    Add documents to the vector store in an idempotent manner using unique IDs.

    Args:
        vector_store: Chroma vector store instance.
        documents: List of Document objects to add.
        batch_size: Number of documents to process in each batch.

    Returns:
        Number of documents successfully added.
    """
    if not documents:
        logger.warning("No documents provided to add to vector store.")
        return 0

    # Get initial document count to track what's actually new
    initial_count = vector_store._collection.count()
    logger.info(
        "Starting ingestion: %d documents currently in vector store, attempting to add %d documents",
        initial_count,
        len(documents),
    )

    # Enrich documents with embedding model metadata
    enriched_documents = []
    document_ids = []

    for doc in documents:
        # Create unique ID for the document
        doc_id = create_document_id(doc)
        document_ids.append(doc_id)

        # Enrich metadata with embedding information
        enriched_metadata = dict(doc.metadata)
        enriched_metadata.update(
            {
                "embedding_model": "ollama_local",
                "document_id": doc_id,
                "collection_name": COLLECTION_NAME,
            }
        )

        # Create new document with enriched metadata
        enriched_doc = Document(
            page_content=doc.page_content,
            metadata=enriched_metadata,
        )
        enriched_documents.append(enriched_doc)

    added_count = 0

    # Process documents in batches to avoid memory issues
    for i in range(0, len(enriched_documents), batch_size):
        batch_docs = enriched_documents[i : i + batch_size]
        batch_ids = document_ids[i : i + batch_size]

        try:
            # Add documents with IDs to prevent duplicates
            vector_store.add_documents(
                documents=batch_docs,
                ids=batch_ids,
            )

            added_count += len(batch_docs)
            logger.info(
                "Added batch %d-%d (%d documents) to vector store",
                i + 1,
                min(i + batch_size, len(enriched_documents)),
                len(batch_docs),
            )

        except Exception as e:
            logger.error(
                "Error adding batch %d-%d to vector store: %s",
                i + 1,
                min(i + batch_size, len(enriched_documents)),
                e,
                exc_info=True,
            )
            # Continue with next batch instead of failing completely
            continue

    # Get final count to see what actually changed
    final_count = vector_store._collection.count()
    net_added = final_count - initial_count

    if net_added == 0:
        logger.info(
            "Processed %d documents: %d were upserted/replaced (no net increase), final count: %d",
            added_count,
            len(documents),
            final_count,
        )
    else:
        logger.info(
            "Successfully added %d out of %d documents to vector store (net increase: +%d, final count: %d)",
            added_count,
            len(documents),
            net_added,
            final_count,
        )

    return added_count


def get_vector_store_stats(vector_store: Chroma) -> dict:
    """
    Get statistics about the current vector store.

    Args:
        vector_store: Chroma vector store instance.

    Returns:
        Dictionary containing vector store statistics.
    """
    try:
        # Get collection info
        collection = vector_store._collection
        count = collection.count()

        # Get sample of metadata to understand document types
        if count > 0:
            sample_results = collection.peek(limit=min(50, count))
            file_types = set()
            element_types = set()

            if (
                sample_results
                and "metadatas" in sample_results
                and sample_results["metadatas"]
            ):
                for metadata in sample_results["metadatas"]:
                    if metadata:
                        file_type = metadata.get("file_type", "unknown")
                        element_type = metadata.get("element_type", "unknown")
                        if file_type is not None:
                            file_types.add(str(file_type))
                        if element_type is not None:
                            element_types.add(str(element_type))
        else:
            file_types = set()
            element_types = set()

        stats = {
            "total_documents": count,
            "file_types": sorted(list(file_types)),
            "element_types": sorted(list(element_types)),
            "collection_name": COLLECTION_NAME,
            "persist_directory": CHROMA_PERSIST_DIRECTORY,
        }

        return stats

    except Exception as e:
        logger.error("Error getting vector store stats: %s", e, exc_info=True)
        return {
            "total_documents": 0,
            "file_types": [],
            "element_types": [],
            "collection_name": COLLECTION_NAME,
            "persist_directory": CHROMA_PERSIST_DIRECTORY,
            "error": str(e),
        }


def clear_vector_store(vector_store: Optional[Chroma] = None) -> bool:
    """
    Clear all documents from the vector store.

    Args:
        vector_store: Optional Chroma vector store instance. If None, creates a new one.

    Returns:
        True if successful, False otherwise.
    """
    try:
        if vector_store is None:
            vector_store = initialize_vector_store()

        # Get all document IDs and delete them
        collection = vector_store._collection
        count_before = collection.count()

        if count_before > 0:
            # Delete all documents
            collection.delete()
            logger.info("Cleared %d documents from vector store", count_before)
        else:
            logger.info("Vector store was already empty")

        return True

    except Exception as e:
        logger.error("Error clearing vector store: %s", e, exc_info=True)
        return False
