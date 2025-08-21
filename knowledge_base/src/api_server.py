import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel

from .data_embedding import get_vector_store_stats, initialize_vector_store
from .utils.logging_config import setup_logging
from .utils.model_config import ModelConfig, get_optimized_config

setup_logging()
logger = logging.getLogger(__name__)

ConversationHistory = Dict[str, List[Dict[str, str]]]
MessageDict = Dict[str, str]


# TODO: probably move to dependency injection
class AppState:
    def __init__(self) -> None:
        self.vector_store: Optional[Chroma] = None
        self.llm: Optional[BaseLanguageModel] = None
        self.config: Optional[ModelConfig] = None
        # TODO: just for test in-memory
        self.conversation_history: ConversationHistory = {}


APP_STATE = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector store and LLM on startup."""
    try:
        logger.info("Initializing RAG system...")

        APP_STATE.vector_store = initialize_vector_store()
        logger.info("Vector store initialized successfully")

        APP_STATE.config = get_optimized_config()
        APP_STATE.llm = APP_STATE.config.get_llm()
        logger.info("LLM initialized successfully")

        stats = get_vector_store_stats(APP_STATE.vector_store)
        logger.info("Vector store stats: %s", stats)

        yield

    except Exception as e:
        logger.error("Failed to initialize RAG system: %s", e)
        raise


app = FastAPI(lifespan=lifespan)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    """Chat message model for API requests."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    messages: List[ChatMessage]
    session_id: Optional[str] = "default"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]
    session_id: str


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Zestify RAG API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = (
            get_vector_store_stats(APP_STATE.vector_store)
            if APP_STATE.vector_store
            else {}
        )
        return {
            "status": "healthy",
            "vector_store_initialized": APP_STATE.vector_store is not None,
            "llm_initialized": APP_STATE.llm is not None,
            "vector_store_stats": stats,
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "unhealthy", "error": str(e)}


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    if not APP_STATE.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        stats = get_vector_store_stats(APP_STATE.vector_store)
        return {"vector_store_stats": stats}
    except Exception as e:
        logger.error("Error getting stats: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error getting stats: {str(e)}"
        ) from e


def contextualize_question(question: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Contextualize a follow-up question based on chat history.

    Args:
        question: The current user question
        chat_history: List of previous messages

    Returns:
        A contextualized question for better retrieval
    """
    if not chat_history or APP_STATE.llm is None:
        return question

    # If chat history exists, use LLM to contextualize the question
    contextualization_prompt = f"""
Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question that can be understood 
without the conversation history. Do NOT answer the question, just reformulate it.

Chat History:
{chr(10).join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-6:]])}

Follow-up Question: {question}

Standalone Question:"""

    try:
        response = APP_STATE.llm.invoke(contextualization_prompt)
        # Handle different response types from LangChain models
        if hasattr(response, "content") and isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            # Fallback: convert to string
            return str(response).strip()
    except Exception as e:
        logger.warning("Failed to contextualize question: %s", e)
        return question


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for RAG queries with conversation history.

    Args:
        request: Chat request containing messages, session ID, and optional parameters.

    Returns:
        ChatResponse with generated response, sources, and metadata.
    """
    if not APP_STATE.vector_store or not APP_STATE.llm:
        raise HTTPException(
            status_code=503, detail="RAG system not properly initialized"
        )

    try:
        session_id = request.session_id or "default"

        # Get or initialize conversation history for this session
        if session_id not in APP_STATE.conversation_history:
            APP_STATE.conversation_history[session_id] = []

        current_history = APP_STATE.conversation_history[session_id]

        # Add new messages to history
        for message in request.messages:
            if message.role in ["user", "assistant"]:
                current_history.append(
                    {"role": message.role, "content": message.content}
                )

        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400, detail="No user message found in request"
            )

        latest_question = user_messages[-1].content
        logger.info(
            "Processing query for session %s: %s",
            session_id,
            latest_question[:100] + "...",
        )

        # Contextualize the question using conversation history
        # Only use history up to the current question (exclude the current question)
        history_for_context = [
            msg for msg in current_history[:-1] if msg["role"] in ["user", "assistant"]
        ]
        contextualized_question = contextualize_question(
            latest_question, history_for_context
        )

        if contextualized_question != latest_question:
            logger.info(
                "Contextualized question: %s", contextualized_question[:100] + "..."
            )

        relevant_docs = APP_STATE.vector_store.similarity_search(
            contextualized_question,
            k=5,
        )

        if not relevant_docs:
            logger.warning("No relevant documents found for query")
            response = ChatResponse(
                response="I couldn't find any relevant information in the knowledge base for your question.",
                sources=[],
                metadata={
                    "query": latest_question,
                    "contextualized_query": contextualized_question,
                    "retrieved_docs": 0,
                },
                session_id=session_id,
            )
            return response

        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in relevant_docs
            ]
        )

        conversation_context = ""
        if history_for_context:
            conversation_context = "\n\nConversation History:\n" + "\n".join(
                [
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in history_for_context[-4:]  # Include last 4 exchanges
                ]
            )

        prompt_template = f"""
You are an expert assistant for the Zestify project. Use the provided context to answer the question accurately and concisely.

Context:
{context}{conversation_context}

Current Question: {latest_question}

Instructions:
- Answer based on the provided context and conversation history
- If the context doesn't contain enough information, say so clearly
- Be specific and provide examples when possible
- Mention relevant source files when appropriate
- When showing code, format it in markdown code blocks with appropriate language tags (```python, ```javascript, etc.)
- If asked for a specific file, try to return the complete relevant code from the context
- Keep your response concise but informative

Answer:
"""

        logger.info("Generating response with LLM...")
        llm_response = APP_STATE.llm.invoke(prompt_template)

        # Extract response content with proper type handling
        response_content: str
        if hasattr(llm_response, "content"):
            content = llm_response.content
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
        elif isinstance(llm_response, str):
            response_content = llm_response
        else:
            response_content = str(llm_response)

        # Add assistant response to conversation history
        current_history.append({"role": "assistant", "content": response_content})

        # Limit conversation history to prevent memory bloat
        if len(current_history) > 20:  # Keep last 20 messages
            APP_STATE.conversation_history[session_id] = current_history[-20:]

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

        logger.info("Response generated successfully for session %s", session_id)

        return ChatResponse(
            response=response_content,
            sources=sources,
            metadata={
                "query": latest_question,
                "contextualized_query": contextualized_question,
                "retrieved_docs": len(relevant_docs),
                "total_context_length": len(context),
                "conversation_length": len(current_history),
                "model_used": "optimized_config",
            },
            session_id=session_id,
        )

    except Exception as e:
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
