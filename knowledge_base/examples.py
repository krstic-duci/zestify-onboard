import os

# TODO: delete google-genai libs later
# from google import genai
# from google.genai.types import EmbedContentConfig, EmbedContentResponse
from typing import Dict

from bs4.filter import SoupStrainer
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from typing_extensions import List, TypedDict

# https://python.langchain.com/docs/tutorials/chatbot/
# https://python.langchain.com/docs/tutorials/rag/
# https://python.langchain.com/docs/tutorials/qa_chat_history/
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

llm = init_chat_model(
    "gemini-1.5-flash",
    model_provider="google_genai",
    google_api_key=SecretStr(gemini_api_key),
)
# print(llm.invoke("What's the capital of France?"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=SecretStr(gemini_api_key)
)


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
_ = vector_store.add_documents(documents=all_splits)

_template = """
You are an expert assistant. Use the provided context to answer the question concisely.

Context:
{context}

Question:
{question}

Answer:
"""


class LocalPrompt:
    """Minimal prompt shim with an invoke(inputs) method compatible with the
    existing `generate` function. It returns a list-of-messages that `llm.invoke`
    accepts (a single user message).
    """

    def __init__(self, template: str) -> None:
        self.template = template

    def invoke(self, inputs: Dict[str, str]):
        text = self.template.format(**inputs)
        return [{"role": "user", "content": text}]


prompt = LocalPrompt(_template)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


state: State = {
    "question": "What is Self-Reflection",
    "context": [],
    "answer": "",
}
state["context"] = retrieve(state)["context"]
answer = generate(state)["answer"]
if isinstance(answer, list):
    answer = " ".join(str(item) for item in answer)
state["answer"] = answer
print(state["answer"])
