## Project Overview

This project is designed to create a sophisticated, AI-powered onboarding assistant. The system will provide a chat-based interface for new users to ask questions and receive informative answers about a specific GitHub repository. This approach facilitates a smoother and more efficient onboarding process by leveraging a Retrieval-Augmented Generation (RAG) architecture, with a clear distinction between the data ingestion and the user-facing chat components.

The entire technology stack is built upon free and open-source software, with the Large Language Model (LLM) accessed via the Gemini API to accommodate local hardware constraints.

---

## Phase 1: The Knowledge base Service (Python)

This phase is responsible for processing the knowledge source (the GitHub repository) and preparing it for the AI. This is an offline process, executed via a local Python script.

### Technology Stack

- **Orchestration:** Python with the LangChain framework.
- **Vector Database:** ChromaDB, running locally for ease of use and persistence.
- **Embedding Model:** Google's `text-embedding-004`, accessed through the Gemini API.
- **Google AI lib**: Google's `google-genai` a Python free lib is to be using for Gemini API.
- **Vercel AI SDK**: Vercel's AI SDK will be used for building the chat interface and Next.js > 15.x.x

### Python Development Environment

- **Dependency Management:** `uv`. All dependencies will be managed in `pyproject.toml` and locked with `uv.lock`. The command `uv sync --frozen` will be used to install the exact versions of dependencies.

* **Running Python:** `uv run {name}.py` will be used to run the script.

- **Linting & Formatting:** `ruff` will be used for code linting and formatting to ensure consistency.
- **Type Checking:** `mypy` will be used for static type checking to maintain code quality.
- **Testing:** `pytest` will be used for writing and running unit tests.

### Workflow

1.  **Load Data:** A Python script (`ingest.py`) will use LangChain's `GitLoader` to clone the target GitHub repository and load all relevant source files.
2.  **Split Documents:** The loaded documents will be passed through a `RecursiveCharacterTextSplitter` in LangChain to break them into smaller, semantically coherent chunks.
3.  **Generate Embeddings:** For each text chunk, the script will call the Gemini API to generate a vector embedding.
4.  **Store in Vector DB:** The script will store each chunk and its corresponding vector in a local ChromaDB database.

---

## Phase 2: The Chat Interface (Next.js)

This phase is the user-facing application. It will be a web-based interface built with modern frontend technologies.

### Technology Stack

- **UI Framework:** Vercel's AI SDK, implemented with Next.js and React.
- **Backend Logic:** Next.js API routes will serve as the lightweight backend.
- **LLM API:** Gemini API (for generating chat responses).
- **Vector Database:** The same ChromaDB instance created in Phase 1.

### Node.js Development Environment

- **Dependency Management:** ALWAYS `pnpm`
- **Linting & Formatting:** `ESLint` and `Prettier`.
- **Testing:** `Jest` or `Vitest`.

### Workflow

1.  **User Interaction:** A user types a question into the chat interface.
2.  **API Request:** The frontend sends the question to a Next.js API route.
3.  **Retrieval & Generation:** The backend retrieves relevant chunks from ChromaDB, constructs a prompt, and calls the Gemini API to get a response.
4.  **Stream Response:** The final answer is streamed back to the UI.

---

## Collaboration and Coding Standards

To ensure the project is easy to understand and maintain, we will adhere to the following standards across both Python and TypeScript codebases:

- **Docstrings/JSDoc:** Every method, function, and class must have a clear and concise docstring/JSDoc explaining its purpose, arguments, and return values.
- **No Magic Numbers:** Avoid using unexplained numbers directly in the code. Declare them as constants with descriptive names.
- **Clear Naming:** Use descriptive and unambiguous names for variables, functions, and classes.
- **Comments for Clarity:** Add comments to explain the "why" behind complex or non-obvious code, not just the "what."
- **Ruff sort**: Every file MUST be sorted accorting to Ruff rules located in `pyproject.toml` file.
