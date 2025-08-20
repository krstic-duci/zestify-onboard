from typing import Dict, List, TypedDict

from bs4.filter import SoupStrainer
from knowledge_base.src.utils.model_config import get_optimized_config
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# https://python.langchain.com/docs/tutorials/chatbot/
# https://python.langchain.com/docs/tutorials/rag/
# https://python.langchain.com/docs/tutorials/qa_chat_history/
config = get_optimized_config()
llm = config.get_llm()
print(llm.invoke("What's the capital of France?"))

embeddings = config.get_embeddings()


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
    collection_name="example_collection_db",
    embedding_function=embeddings,
    persist_directory="./examples_chroma_langchain_db",
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
# print(state["answer"])
#
########################################################################################
#                     BELOW ARE EXAMPLES OF HOW FUNCTIONS WORKS                        #
########################################################################################

# ------------------------------------------------------------------------------------ #
# 1. `clean_document`
# ------------------------------------------------------------------------------------ #
# Quick explanation: how `clean_python_file` and `PythonCodeVisitor` work (step-by-step)
# 1) `clean_python_file(content, file_path)` is the entry point.
#    - If the file is empty it returns an empty list.
#    - It calls `ast.parse(content)` to get a syntax tree (AST) for the file.
#    - It creates `PythonCodeVisitor(content, file_path)` and calls `visitor.visit(tree)`.
#    - Finally it returns `visitor.documents` (a list of `Document` objects).
#
# 2) `PythonCodeVisitor` is an `ast.NodeVisitor` specialized to extract only the
# meaningful code blocks (classes and functions). On init it:
#    - saves `source_lines = content.splitlines(keepends=True)` so original
#      newlines and indentation are preserved when slicing the text,
#    - pre-allocates `self.documents = []` to collect results,
#    - keeps `self.current_class` to build qualified names for methods.
#
# 3) When the visitor encounters a `ClassDef` node it:
#    - records the class name in `self.current_class`,
#    - extracts the docstring via `ast.get_docstring(node)`,
#    - calls `_get_source_segment(node)` to reconstruct the exact source text
#      for the node using AST line/column offsets, and
#    - builds a `Document(page_content=..., metadata={...})` and appends it.
#    - it then `generic_visit(node)` so methods inside the class are visited
#      and emitted as separate `Document`s (with `element_name` like
#      `ClassName.method`). Finally it restores the previous class context.
#
# 4) When the visitor encounters a `FunctionDef` node it:
#    - skips private top-level functions (names starting with `_`) unless they
#      are methods inside a class,
#    - extracts the docstring and source text using `_get_source_segment`,
#    - creates a `Document` with a header, docstring (if present) and a
#      fenced-code `Source Code` block, and appends it to `self.documents`.
#
# 5) `_get_source_segment(node)` (the helper):
#    - converts AST 1-based `lineno` to 0-based indexes for `source_lines`,
#    - uses `col_offset` and `end_col_offset` to slice text precisely,
#    - fast path: if start and end are on the same line, return that slice,
#    - slow path: if the node spans multiple lines, build `first_line`,
#      `inner_lines` and `last_line` and join them to preserve exact text
#      (including indentation and newline characters).
#
#
# 6) Example:
# method: clean_document > "clean_python_file"
# file: /src/tmp/dummy.py
# """A dummy module."""
#
# def top_level_func(x):
#     """A top level function."""
#     return x + 1
#
# class MyClass:
#     """A dummy class."""
#     def __init__(self, y):
#         self.y = y
#
#     def method(self, z):
#         """A method."""
#         return self.y + z
#
# # When clean_document processes this file, it will parse the code and return a list of
# three `Document` objects:
# [
#     # Document 1: The top-level function
#     Document(
#         page_content=(
#             "--- Function: top_level_func ---\n"
#             "Docstring:\nA top level function.\n"
#             "Source Code:\n"
#             "```python\n"
#             "def top_level_func(x):\n"
#             '    """A top level function."""\n'
#             "    return x + 1\n"
#             "```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/dummy.py",
#             "element_type": "function",
#             "element_name": "top_level_func",
#             "start_line": 5,
#             "end_line": 7,
#         },
#     ),
#
#     # Document 2: The class definition
#     Document(
#         page_content=(
#             "--- Class: MyClass ---\n"
#             "Docstring:\nA dummy class.\n"
#             "Source Code:\n"
#             "```python\n"
#             "class MyClass:\n"
#             '    """A dummy class."""\n'
#             "    def __init__(self, y):\n"
#             "        self.y = y\n\n"
#             "    def method(self, z):\n"
#             '        """A method."""\n'
#             "        return self.y + z\n"
#             "```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/dummy.py",
#             "element_type": "class",
#             "element_name": "MyClass",
#             "start_line": 9,
#             "end_line": 17,
#         },
#     ),
#
#     # Document 3: The method inside the class
#     Document(
#         page_content=(
#             "--- Function: MyClass.method ---\n"
#             "Docstring:\nA method.\n"
#             "Source Code:\n"
#             "```python\n"
#             "def method(self, z):\n"
#             '    """A method."""\n'
#             "    return self.y + z\n"
#             "```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/dummy.py",
#             "element_type": "function",
#             "element_name": "MyClass.method",
#             "start_line": 15,
#             "end_line": 17,
#         },
#     ),
# ]
#
# ------------------------------------------------------------------------------------ #
# method: clean_document > "clean_js_file" or "clean_html_file"
# file: /src/tmp/dummy.js                                                                                                    │
# function greet(name) {
#   console.log("Logging the name");
#   return `Hello, ${name}!`;
# }
#
# const PI = 3.14;
#
# When `clean_document` processes other files than Python, it will perform a basic
# cleaning (Jinja gets converted to MD) and return a list with a SINGLE `Document`
# object:
#
# [
#     Document(
#         page_content=(
#             "function greet(name) {\n"
#             "  \n"
#             "  return `Hello, ${name}!`;\n"
#             "}\n\n"
#             "const PI = 3.14;"
#         ),
#         metadata={
#             "source": "/src/tmp/dummy.js",
#         },
#     ),
# ]
#
#
#
# ------------------------------------------------------------------------------------ #
# 2. `data_split`
# ------------------------------------------------------------------------------------ #
# Quick explanation: how `split_documents(documents)` works (step-by-step)
# 1) Purpose:
#    - `split_documents` receives a list (or single) `Document` object(s) and its job is
#      to produce a new list of smaller, semantically-meaningful chunks that are ready
#      to be embedded and indexed in a vector store.
#
# 2) Key constants and splitters:
#    - `CHUNK_SIZE` and `CHUNK_OVERLAP` control the target length and
#      overlap for text chunks (default in this module: 1000/200 characters).
#    - `python_splitter` and `js_splitter` are `RecursiveCharacterTextSplitter`
#      instances created with language-specific heuristics using
#      `from_language(language=Language.PYTHON/JS, ...)`.
#    - `general_splitter` is a fallback splitter used for HTML/Jinja/Markdown and
#      other text files.
#
# 3) High-level flow inside `split_documents`:
#    - For each input `Document`:
#      a) It reads the source path and file suffix (to choose split strategy).
#      b) Calls `clean_document(doc)` which may return multiple `Document`s
#         (notably Python files return one Document per function/class).
#      c) For each cleaned Document:
#         - If `element_type` is `function` or `class` we keep that Document
#           as a single chunk (do not further split). We enrich its metadata
#           with `file_type` and `char_length` and append it to the output.
#         - Otherwise we select the appropriate splitter (python/js/general)
#           and call `splitter.split_documents([cleaned_doc])` to produce
#           multiple smaller chunks.
#         - Each produced chunk is enriched with `file_type` and `char_length`.
#           If `element_type` is missing it is set to `chunk` so downstream
#           code knows this came from splitting.
#    - Errors while cleaning/splitting are caught and logged; the function
#      continues processing other documents.
#    - The function logs the total documents processed and chunks produced,
#      then returns the `all_chunks` list.
#
# 4) Example:
#
# method: data_split > "split_documents"
# Imagine we have these Documents from the loader:
# input_docs = [
#     A large README file that needs chunking
#     Document(
#         page_content=(
#             "# MyProject\n\n"
#             "This is a comprehensive guide to using MyProject. It provides detailed documentation "
#             "on installation, configuration, and usage patterns. The project was created to solve "
#             "common problems in data processing workflows...\n\n"
#             "## Installation\n\n"
#             "To install the project, you need Python 3.8+ and the following dependencies:\n"
#             "- pandas>=1.3.0\n- numpy>=1.20.0\n- requests>=2.25.0\n\n"
#             "Run the following command:\n```bash\npip install myproject\n```\n\n"
#             "## Configuration\n\n"
#             "Create a config.yaml file with the following structure...\n"
#             # Imagine this continues for 2500+ characters
#         ),
#         metadata={"source": "/src/tmp/README.md"},
#     ),
#     A Python file with multiple functions
#     Document(
#         page_content=(
#             "def calculate_average(numbers):\n"
#             '    """Calculate the average of a list of numbers."""\n'
#             "    return sum(numbers) / len(numbers)\n\n"
#             "class DataProcessor:\n"
#             '    """Process data efficiently."""\n'
#             "    def __init__(self, config):\n"
#             "        self.config = config\n\n"
#             "    def process(self, data):\n"
#             '        """Process the input data."""\n'
#             "        return data.upper()\n"
#         ),
#         metadata={"source": "/src/tmp/utils.py"},
#     ),
# ]

# After calling split_documents(input_docs), here's what we get:
# output_chunks = [
#     # Python function (kept as single chunk, not split further)
#     Document(
#         page_content=(
#             "--- Function: calculate_average ---\n"
#             "Docstring:\nCalculate the average of a list of numbers.\n"
#             "Source Code:\n"
#             "```python\n"
#             "def calculate_average(numbers):\n"
#             '    """Calculate the average of a list of numbers."""\n'
#             "    return sum(numbers) / len(numbers)\n"
#             "```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/utils.py",
#             "element_type": "function",
#             "element_name": "calculate_average",
#             "start_line": 1,
#             "end_line": 3,
#             "file_type": "py",
#             "char_length": 178,
#         },
#     ),
#     # Python class (kept as single chunk, not split further)
#     Document(
#         page_content=(
#             "--- Class: DataProcessor ---\n"
#             "Docstring:\nProcess data efficiently.\n"
#             "Source Code:\n"
#             "```python\n"
#             "class DataProcessor:\n"
#             '    """Process data efficiently."""\n'
#             "    def __init__(self, config):\n"
#             "        self.config = config\n\n"
#             "    def process(self, data):\n"
#             '        """Process the input data."""\n'
#             "        return data.upper()\n"
#             "```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/utils.py",
#             "element_type": "class",
#             "element_name": "DataProcessor",
#             "start_line": 5,
#             "end_line": 12,
#             "file_type": "py",
#             "char_length": 267,
#         },
#     ),
#     # Method inside the class (extracted separately)
#     Document(
#         page_content=(
#             "--- Function: DataProcessor.process ---\n"
#             "Docstring:\nProcess the input data.\n"
#             "Source Code:\n"
#             "```python\n"
#             "def process(self, data):\n"
#             '    """Process the input data."""\n'
#             "    return data.upper()\n"
#             "```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/utils.py",
#             "element_type": "function",
#             "element_name": "DataProcessor.process",
#             "start_line": 9,
#             "end_line": 11,
#             "file_type": "py",
#             "char_length": 156,
#         },
#     ),
#     # README chunk 1 (split because it's too long)
#     Document(
#         page_content=(
#             "# MyProject\n\n"
#             "This is a comprehensive guide to using MyProject. It provides detailed documentation "
#             "on installation, configuration, and usage patterns. The project was created to solve "
#             "common problems in data processing workflows...\n\n"
#             "## Installation\n\n"
#             "To install the project, you need Python 3.8+ and the following dependencies:\n"
#             "- pandas>=1.3.0\n- numpy>=1.20.0\n- requests>=2.25.0\n\n"
#             "Run the following command:\n```bash\npip install myproject\n```\n\n"
#             # This chunk is ~1000 characters
#         ),
#         metadata={
#             "source": "/src/tmp/README.md",
#             "file_type": "md",
#             "element_type": "chunk",
#             "char_length": 987,
#         },
#     ),
#     # README chunk 2 (continuation with 200 char overlap)
#     Document(
#         page_content=(
#             # Last 200 chars from chunk 1 for overlap
#             "Run the following command:\n```bash\npip install myproject\n```\n\n"
#             "## Configuration\n\n"
#             "Create a config.yaml file with the following structure...\n"
#             # Rest of the README content
#         ),
#         metadata={
#             "source": "/src/tmp/README.md",
#             "file_type": "md",
#             "element_type": "chunk",
#             "char_length": 1150,
#         },
#     ),
# ]
#
#
#
# ------------------------------------------------------------------------------------ #
# 3. `data_embedding`
# ------------------------------------------------------------------------------------ #
# Quick explanation: how `data_embedding.py` works (step-by-step)
# 1) Purpose:
#    - Provide a small, focused layer that creates deterministic IDs for
#      Documents, initializes a Chroma vector store (persisted to disk), adds
#      documents in an idempotent/batched way, reports simple stats, and can
#      clear the collection when needed.
#
# 2) Key functions and responsibilities:
#    - `create_document_id(doc: Document) -> str`:
#      * Inputs: a `Document` with `page_content` and `metadata`.
#      * Output: a deterministic SHA-256 hex string based on the document's
#        source path, element type/name, start line and full page content.
#      * Why: ensures the same logical document always maps to the same ID so
#        repeated indexing attempts don't create duplicates.
#
#    - `initialize_vector_store() -> Chroma`:
#      * Inputs: none (loads configuration internally).
#      * Behavior: imports the repository's model/config helper to obtain an
#        embeddings object, creates the local persist directory (if missing),
#        and constructs a `Chroma` instance bound to `COLLECTION_NAME`.
#      * Output: a ready-to-use `Chroma` vector store instance using the
#        project's embedding callable.
#      * Notes: designed to prefer local/hybrid embeddings.
#
#    - `add_documents_to_store(vector_store: Chroma, documents: List[Document], batch_size: int = 100) -> int`:
#      * Inputs: a Chroma store, a list of LangChain `Document`s and an optional
#        batch size for memory control.
#      * Behavior: builds a stable ID for each document using
#        `create_document_id`, enriches the metadata with embedding model and
#        collection information, and calls `vector_store.add_documents`
#        in batches. Each batch uses the precomputed IDs so repeated calls are
#        idempotent (duplicates are avoided).
#      * Output: number of documents successfully added.
#      * Failure mode: exceptions during a batch are logged and the routine
#        continues with subsequent batches (best-effort indexing).
#
#    - `get_vector_store_stats(vector_store: Chroma) -> dict`:
#      * Inputs: a Chroma store.
#      * Behavior: queries the underlying collection for a count, peeks a small
#        sample of metadata entries (up to 10) to collect file/element types,
#        and returns a concise stats dictionary.
#      * Output: dict with `total_documents`, `file_types`, `element_types`,
#        `collection_name`, and `persist_directory`. On error returns a dict
#        with `error` and empty/default fields.
#
#    - `clear_vector_store(vector_store: Optional[Chroma] = None) -> bool`:
#      * Inputs: optional Chroma instance (if omitted the function calls
#        `initialize_vector_store`).
#      * Behavior: counts documents and calls `collection.delete()` to wipe the
#        store when non-empty. Returns True on success, False on failure.
#      * Notes: helpful for tests, development, or rebuilding the index from
#        scratch.
#
# 3) Example:
#
# method: data_embedding > "initialize_vector_store" + "add_documents_to_store"
#
# Step 1: Initialize the vector store
# vector_store = initialize_vector_store()
# Creates: Chroma(collection_name="rag_chroma_db", persist_directory="./chroma_langchain_db")
#
# Step 2: Prepare documents to index (these come from data_split output)
# docs_to_index = [
#     Document(
#         page_content=(
#             "--- Function: calculate_sum ---\n"
#             "Docstring:\nCalculate sum of two numbers.\n"
#             "Source Code:\n```python\n"
#             "def calculate_sum(a, b):\n"
#             "    return a + b\n```\n"
#         ),
#         metadata={
#             "source": "/src/tmp/math_utils.py",
#             "element_type": "function",
#             "element_name": "calculate_sum",
#             "file_type": "py",
#             "char_length": 123
#         }
#     ),
#     Document(
#         page_content="# Installation Guide\nTo install this package, run pip install...",
#         metadata={
#             "source": "/src/tmp/README.md",
#             "file_type": "md",
#             "element_type": "chunk",
#             "char_length": 67
#         }
#     )
# ]
#
# Step 3: Add documents (with automatic ID generation)
# Internally, create_document_id() generates deterministic IDs:
#  - For function doc: sha256("/src/tmp/math_utils.py:function:calculate_sum:1:5:--- Function: calculate_sum ---...")
#    Result ID: "a7f2c8d9e1b4f3a6c2d8e9f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"
#  - For README chunk: sha256("/src/tmp/README.md::::# Installation Guide...")
#    Result ID: "b2f8d1a3c7e9f2a4b6d0c8e1f3a5b7d9c2e4f6a8b0d2e4f6a8b0c2e4f6a8b0c2"
#
# added_count = add_documents_to_store(vector_store, docs_to_index)
# Logs: "Added 2 documents to vector store successfully"
# Returns: 2
#
# Step 4: Check statistics
# stats = get_vector_store_stats(vector_store)
# print(stats)
#  Output:
# {
#     'total_documents': 2,
#     'file_types': ['py', 'md'],
#     'element_types': ['function', 'chunk'],
#     'collection_name': 'rag_chroma_db',
#     'persist_directory': './chroma_langchain_db'
# }
#
# Step 5: Re-run with same documents (idempotency test)
# added_again = add_documents_to_store(vector_store, docs_to_index)
# Logs: "Documents with existing IDs were upserted (updated): 2"
# Logs: "Net new documents added: 0"
# Returns: 0 (no new documents, just updates)
#
# Step 6: Add a modified version of the same function
# modified_doc = Document(
#     page_content=(
#         "--- Function: calculate_sum ---\n"
#         "Docstring:\nCalculate sum of two integers.\n"  # Changed docstring
#         "Source Code:\n```python\n"
#         "def calculate_sum(a, b):\n"
#         "    \"\"\"Calculate sum of two integers.\"\"\"\n"  # Added docstring
#         "    return a + b\n```\n"
#     ),
#     metadata={
#         "source": "/src/tmp/math_utils.py",
#         "element_type": "function",
#         "element_name": "calculate_sum",
#         "file_type": "py",
#         "char_length": 156  # Different length due to content change
#     }
# )
#
# added_modified = add_documents_to_store(vector_store, [modified_doc])
# Creates new ID: "c3a9e2b5f8d1a4c7e0b3f6a9c2e5f8b1d4a7c0e3f6a9b2e5f8a1d4b7c0e3f6a9"
# Logs: "Added 1 documents to vector store successfully"
# Returns: 1 (new document because content changed)
#
# final_stats = get_vector_store_stats(vector_store)
# Output:
# {
#     'total_documents': 3,  # Original 2 + 1 modified version
#     'file_types': ['py', 'md'],
#     'element_types': ['function', 'chunk'],
#     'collection_name': 'rag_chroma_db',
#     'persist_directory': './chroma_langchain_db'
# }
#
# 4) Edge cases and implementation notes:
#    - Deterministic IDs: the `create_document_id` function includes the
#      document's full `page_content` in the hash. If the content changes the
#      ID changes as well — this is intentional so edits create a new entry.
#    - Large document lists: `add_documents_to_store` uses batching to avoid
#      memory spikes; tune `batch_size` to suit available RAM on your machine.
#    - Partial failures: the function logs and skips failing batches instead
#      of raising, favoring resilience during large indexing runs. If you need
#      transactional behavior raise errors instead.
#    - Direct collection access: some helpers use `vector_store._collection` to
#      call Chroma internals (count/peek/delete). That relies on the specific
#      Chroma client used; if you swap vector backends you will need to adapt
#      these calls.
#
# 5) Why this module exists:
#    - Keeps embedding and persistence logic in one place so the rest of the
#      ingestion pipeline (`clean_document`, `data_split`) can remain focused
#      on producing high-quality Document objects. This module handles the
#      idempotency, batching, and simple diagnostics needed to keep a local
#      Chroma index healthy and debuggable.
