### Phase 1: The Knowledge Base Service (Python Ingestion Pipeline)

This phase is all about creating the "brain" of your AI assistant. It's an offline process that you will run to populate your ChromaDB vector store.

1.  **Loading the Data**
    1.  Use LangChain's `GitLoader` to clone the `https://github.com/krstic-duci/zestify` repository into a temporary directory.
    2.  The `GitLoader` will load all the files from the repository into a list of `Document` objects.
2.  **Data Cleaning and Pre-processing**
    1.  For each loaded `Document`, identify the file type (e.g., `.py`, `.js`, `.html`).
    2.  Implement a data cleaning function for each file type:
        1.  For Python files, consider using the `ast` module to extract functions, classes, and docstrings.
        2.  For HTML/Jinja files, use `markdownify` to strip out HTML tags.
        3.  For JavaScript files, use regular expressions or a JS parser to extract meaningful content.
3.  **Splitting the Documents**

    1.  Use a specialized `RecursiveCharacterTextSplitter` for your Python and JavaScript files to create more semantically meaningful chunks (import `Language` from LangChain).
    2.  Use the same `RecursiveCharacterTextSplitter` for your HTML/Jinja files and any other text-based files (without the `Language` import).

4.  **Metadata Enrichment**

    1.  For each chunk (which is a `Document` object), enrich its `metadata` dictionary with useful information:
        1.  `file_path`: The original path of the file in the repository.
        2.  `file_type`: "python", "javascript", "html", etc.
        3.  `element_type`: "function", "class", "docstring", etc.
        4.  `element_name`: The name of the function or class, if applicable.

5.  **Embedding and Storing**

    1.  Initialize the `GoogleGenerativeAIEmbeddings` model (`text-embedding-004`).
    2.  Initialize your `Chroma` vector store.
    3.  Implement an idempotent way to add your documents to the vector store. This involves:
        1.  Creating a unique ID for each `Document` (e.g., by hashing the file path and content).
        2.  Using the `ids` parameter in the `add_documents` method to prevent duplicates.
    4.  Add all your processed and enriched `Document` objects to ChromaDB.

6.  **Evaluation with Ragas**

    1.  Utilize the Ragas framework to evaluate the quality of the RAG pipeline.
    2.  Define a set of test questions and expected answers (ground truth).
    3.  Measure key metrics such as Faithfulness, Answer Relevance, Context Recall, and Context Precision to assess chunking strategy and overall retrieval effectiveness.

7.  **Automation and Error Handling**
    1.  Wrap your ingestion logic in `try...except` blocks to handle potential errors (e.g., invalid code, network issues).
    2.  Add logging to your script to track the progress and any errors.
    3.  (Future) Set up a GitHub Action to automatically run your ingestion script whenever the `zestify` repository is updated.

### Phase 2: The interface service with Vercel AI SDK

    **Questions:**
    1. How to interact with Python service? Include FastAPI and expose RAG via endpoints???
