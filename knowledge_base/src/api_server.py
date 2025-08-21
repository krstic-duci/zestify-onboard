import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .data_embedding import get_vector_store_stats, initialize_vector_store
from .utils.logging_config import setup_logging
from .utils.model_config import get_optimized_config

setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Zestify RAG API",
    description="API for querying the Zestify knowledge base using RAG",
    version="0.1.0",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables for vector store and LLM
vector_store = None
llm = None
config = None


class ChatMessage(BaseModel):
    """Chat message model for API requests."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize vector store and LLM on startup."""
    global vector_store, llm, config

    try:
        logger.info("Initializing RAG system...")

        # Initialize vector store
        vector_store = initialize_vector_store()
        logger.info("Vector store initialized successfully")

        # Initialize LLM configuration
        config = get_optimized_config()
        llm = config.get_llm()
        logger.info("LLM initialized successfully")

        # Log vector store stats
        stats = get_vector_store_stats(vector_store)
        logger.info("Vector store stats: %s", stats)

    except Exception as e:
        logger.error("Failed to initialize RAG system: %s", e)
        raise


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Zestify RAG API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = get_vector_store_stats(vector_store) if vector_store else {}
        return {
            "status": "healthy",
            "vector_store_initialized": vector_store is not None,
            "llm_initialized": llm is not None,
            "vector_store_stats": stats,
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "unhealthy", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for RAG queries.

    Args:
        request: Chat request containing messages and optional parameters.

    Returns:
        ChatResponse with generated response, sources, and metadata.
    """
    if not vector_store or not llm:
        raise HTTPException(
            status_code=503, detail="RAG system not properly initialized"
        )

    try:
        # Extract the latest user message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400, detail="No user message found in request"
            )

        latest_question = user_messages[-1].content
        logger.info("Processing query: %s", latest_question[:100] + "...")

        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(
            latest_question,
            k=5,  # Retrieve top 5 most similar documents
        )

        if not relevant_docs:
            logger.warning("No relevant documents found for query")
            return ChatResponse(
                response="I couldn't find any relevant information in the knowledge base for your question.",
                sources=[],
                metadata={"query": latest_question, "retrieved_docs": 0},
            )

        # Prepare context from retrieved documents
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in relevant_docs
            ]
        )

        # Create prompt template
        prompt_template = """
You are an expert assistant for the Zestify project. Use the provided context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific and provide examples when possible
- Mention relevant source files when appropriate

Answer:
"""

        # Format the prompt
        formatted_prompt = prompt_template.format(
            context=context, question=latest_question
        )

        # Generate response using LLM
        logger.info("Generating response with LLM...")
        response = llm.invoke(formatted_prompt)

        # Extract response content
        if hasattr(response, "content"):
            content = response.content
            # Handle different content types from LLM response
            if isinstance(content, str):
                response_content = content
            elif isinstance(content, list):
                # If it's a list, join the items as strings
                response_content = " ".join(str(item) for item in content)
            elif isinstance(content, dict):
                # If it's a dict, try to extract text or convert to string
                response_content = content.get("text", str(content))
            else:
                response_content = str(content)
        else:
            response_content = str(response)

        # Prepare sources information
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "element_type": doc.metadata.get("element_type", "unknown"),
                "element_name": doc.metadata.get("element_name", ""),
                "preview": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
            }
            for doc in relevant_docs
        ]

        logger.info("Response generated successfully")

        return ChatResponse(
            response=response_content,
            sources=sources,
            metadata={
                "query": latest_question,
                "retrieved_docs": len(relevant_docs),
                "total_context_length": len(context),
                "model_used": "optimized_config",
            },
        )

    except Exception as e:
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        stats = get_vector_store_stats(vector_store)
        return {"vector_store_stats": stats}
    except Exception as e:
        logger.error("Error getting stats: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error getting stats: {str(e)}"
        ) from e
