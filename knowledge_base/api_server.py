import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.data_embedding import get_vector_store_stats, initialize_vector_store
from src.utils.logging_config import setup_logging
from src.utils.model_config import ModelConfig, hybrid_model_config

setup_logging()
logger = logging.getLogger(__name__)

ConversationHistory = Dict[str, List[Dict[str, str]]]
MessageDict = Dict[str, str]


# TODO: probably move to dependency injection
class AppState:
    def __init__(self) -> None:
        self.vector_store: Optional[Chroma] = None
        self.model_config: Optional[ModelConfig] = None
        self.gemini_llm: Optional[ChatGoogleGenerativeAI] = None
        # TODO: in-memory just for test
        self.conversation_history: ConversationHistory = {}


APP_STATE = AppState()
TOP_K = 5
TEMP = 0.7
MAX_TOKENS = 1000


# TODO: schemas folder please
class ChatMessage(BaseModel):
    """Chat message model for API requests."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    messages: List[ChatMessage]
    session_id: Optional[str] = "default"
    max_tokens: Optional[int] = MAX_TOKENS
    temperature: Optional[float] = TEMP


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector store and LLM on startup."""
    try:
        APP_STATE.vector_store = initialize_vector_store()
        logger.info("Vector store initialized successfully")

        APP_STATE.model_config = hybrid_model_config()
        APP_STATE.gemini_llm = APP_STATE.model_config.get_gemini_llm()
        logger.info("LLM initialized successfully")

        yield

    except Exception as e:
        logger.error("Failed to initialize RAG system: %s", e)
        raise


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint for health check."""
    try:
        stats = (
            get_vector_store_stats(APP_STATE.vector_store)
            if APP_STATE.vector_store
            else {}
        )
        return {
            "status": "healthy",
            "vector_store_initialized": APP_STATE.vector_store is not None,
            "llm_initialized": APP_STATE.gemini_llm is not None,
            "vector_store_stats": stats,
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "unhealthy", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for RAG queries with conversation history.

    Args:
        request: Chat request containing messages, session ID, and optional parameters.

    Returns:
        ChatResponse with generated response, sources, and metadata.
    """
    if not APP_STATE.vector_store or not APP_STATE.gemini_llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not properly initialized",
        )

    def contextualize_question(
        question: str, chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Contextualize a follow-up question based on chat history.

        Args:
            question: The current user question
            chat_history: List of previous messages

        Example:
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny."},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome."}
        ]
        Result (multi-line string):
            user: Hi
            assistant: Hello! How can I help?
            user: What's the weather?
            assistant: It's sunny.
            user: Thanks!
            assistant: You're welcome.

        Returns:
            A contextualized question for better retrieval
        """
        if not chat_history or APP_STATE.gemini_llm is None:
            return question

        contextualization_prompt = f"""
    Given the following conversation history and a follow-up question, 
    rephrase the follow-up question to be a standalone question that can be understood 
    without the conversation history. Do NOT answer the question, just reformulate it.

    Chat History:
    {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-6:]])}

    Follow-up Question: {question}

    Standalone Question:"""

        try:
            response = APP_STATE.gemini_llm.invoke(contextualization_prompt)
            if hasattr(response, "content") and isinstance(response.content, str):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        except Exception as e:
            logger.warning("Failed to contextualize question: %s", e)
            return question

    try:
        session_id = request.session_id or "default"

        if session_id not in APP_STATE.conversation_history:
            APP_STATE.conversation_history[session_id] = []

        current_history = APP_STATE.conversation_history[session_id]

        for message in request.messages:
            if message.role in ["user", "assistant"]:
                current_history.append(
                    {"role": message.role, "content": message.content}
                )

        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in request",
            )

        latest_question = user_messages[-1].content
        logger.info(
            "Processing query for session %s: %s",
            session_id,
            latest_question[:100] + "...",
        )

        history_for_context = [
            msg for msg in current_history[:-1] if msg["role"] in ["user", "assistant"]
        ]
        contextualized_question = contextualize_question(
            latest_question, history_for_context
        )

        relevant_docs = APP_STATE.vector_store.similarity_search(
            contextualized_question,
            k=TOP_K,  # TODO: this needs to be tested
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
                    for msg in history_for_context[-4:]
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
        llm_response = APP_STATE.gemini_llm.invoke(prompt_template)

        response_content: str
        if hasattr(llm_response, "content"):
            content = llm_response.content
            if isinstance(content, str):
                response_content = content
            elif isinstance(content, list):
                response_content = " ".join(str(item) for item in content)
            elif isinstance(content, dict):
                response_content = content.get("text", str(content))
            else:
                response_content = str(content)
        elif isinstance(llm_response, str):
            response_content = llm_response
        else:
            response_content = str(llm_response)

        current_history.append({"role": "assistant", "content": response_content})

        # Limit conversation history to prevent memory bloat
        if len(current_history) > 20:
            APP_STATE.conversation_history[session_id] = current_history[-20:]

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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        ) from e
