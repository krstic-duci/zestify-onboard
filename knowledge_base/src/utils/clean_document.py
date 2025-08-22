import ast
import re
import textwrap
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from markdownify import markdownify  # type:ignore

# Type alias for AST nodes that have position information
ASTNodeWithPos = Union[
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.stmt,
    ast.expr,
]


# https://medium.com/@wshanshan/intro-to-python-ast-module-bbd22cd505f7
# https://docs.python.org/3/library/ast.html#ast.NodeVisitor
class PythonCodeVisitor(ast.NodeVisitor):
    """
    An AST visitor that walks through a Python source code's Abstract Syntax Tree
    to extract structured information, specifically classes and functions.

    For each class and function, it creates a separate `Document` object containing
    the element's source code, docstring, and positional metadata.
    """

    def __init__(self, source_code: str, file_path: str) -> None:
        """
        Initializes the visitor.

        Args:
            source_code: The full source code of the Python file as a string.
            file_path: The path to the source file, used for metadata.
        """
        # Store source lines to easily extract code segments later using line numbers.
        self.source_lines = source_code.splitlines(keepends=True)
        # List to accumulate the Document objects created for each class/function.
        self.documents: List[Document] = []
        # Tracks the name of the current class being visited. Used for nested
        # functions/methods.
        self.current_class: Optional[str] = None
        # Store the file path for metadata in the Document objects.
        self.file_path = file_path

    def _get_source_segment(self, node: ASTNodeWithPos) -> str:
        """
        Extracts the full, original source code of a specific AST node.

        This method uses the line and column offset information from the AST node
        to slice the exact segment from the original source code. This is crucial
        for getting the precise code snippet for a function or class.

        Args:
            node: An AST node that has `lineno` and `col_offset` attributes.
                  `lineno` is the 1-based line number, `col_offset` is the 0-based
                  byte offset of the first token on the line.

        Returns:
            The source code segment corresponding to the node as a string.
        """
        # AST line numbers are 1-based, so convert to 0-based for list indexing.
        start_line, start_col = node.lineno - 1, node.col_offset
        # Use `getattr` for backward compatibility and provide a fallback.
        end_line = getattr(node, "end_lineno", node.lineno) - 1
        # If `end_col_offset` is not available, assume it goes to the end of the line.
        end_col = getattr(node, "end_col_offset", len(self.source_lines[start_line]))

        # Many AST nodes (for example simple functions, single-line
        # assignments, or small expressions) start and end on the same
        # source line. In that common case we can return a single slice of
        # that line using the node's start/end column offsets. This is
        # significantly cheaper and simpler than concatenating multiple
        # line parts below and preserves the exact original formatting
        # (including indentation and any trailing newline characters
        # because we stored `source_lines` with `keepends=True`).
        # Important caveats:
        # - `lineno`/`col_offset` are 1-based / 0-based as returned by the
        #   AST, so we convert `lineno` to a 0-based index for the list.
        # - `end_col_offset` may be missing for some older AST nodes or
        #   Python versions; we use a safe fallback above to the line length.
        # - Column offsets are character offsets on the line; using the
        #   original `source_lines` ensures the slice matches the raw text.
        if start_line == end_line:
            return self.source_lines[start_line][start_col:end_col]

        # For nodes that span multiple lines (common for classes, larger
        # functions or multiline expressions), we build the full source
        # segment from three pieces:
        # 1. `first_line`: the slice of the starting line from the node's
        #    start column to the line end. This preserves any leading
        #    indentation and the exact text from the starting token.
        # 2. `inner_lines`: any full lines between the start and the end
        #    line. These are taken as-is from `source_lines` and include
        #    their original trailing newline characters because we
        #    initialized `source_lines` with `keepends=True`.
        # 3. `last_line`: the slice of the final line up to the node's
        #    end column. This ensures the segment ends exactly where the
        #    AST reports the node finishing.
        #
        # Notes / caveats:
        # - `inner_lines` can be an empty list if the node covers only
        #   two lines (start and end). The join below handles that case
        #   correctly.
        # - Using this three-part approach preserves exact original
        #   formatting and is cheaper than iteratively concatenating
        #   strings in a loop (we build a small list and join it once).
        # - This assumes column offsets map to character indices in the
        #   stored lines (works for typical source files). If the file
        #   contains multi-byte characters, offsets are still valid as
        #   returned by Python's AST on current versions.
        # - If end offsets are missing we used safe fallbacks earlier so
        #   slicing here is robust.
        first_line = self.source_lines[start_line][start_col:]
        inner_lines = self.source_lines[start_line + 1 : end_line]
        last_line = self.source_lines[end_line][:end_col]

        return "".join([first_line] + inner_lines + [last_line])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Called automatically by `ast.NodeVisitor` when a class definition is encountered.
        This method extracts the class's source code and metadata to create a Document.
        """
        parent_class = self.current_class
        self.current_class = node.name

        docstring = ast.get_docstring(node)
        class_source = self._get_source_segment(node)

        content = f"--- Class: {node.name} ---\n"
        if docstring:
            content += f"Docstring:\n{textwrap.dedent(docstring)}\n"
        content += f"Source Code:\n```python\n{class_source}\n```"

        self.documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "element_type": "class",
                    "element_name": node.name,
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                },
            )
        )

        self.generic_visit(node)
        self.current_class = parent_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Called automatically by `ast.NodeVisitor` when a function definition is encountered.
        This method extracts the function's source code and metadata to create a Document.
        """
        # Conventionally, private functions (starting with '_') are often skipped
        # unless they are methods within a class (where `self.current_class` would be set).
        if node.name.startswith("_") and not self.current_class:
            self.generic_visit(node)  # Still visit children in case of nested functions
            return

        func_source = self._get_source_segment(node)
        docstring = ast.get_docstring(node)

        # Create a fully qualified name for methods (e.g., "ClassName.method_name").
        prefix = f"{self.current_class}." if self.current_class else ""
        content = f"--- Function: {prefix}{node.name} ---\n"
        if docstring:
            content += f"Docstring:\n{textwrap.dedent(docstring)}\n"
        content += f"Source Code:\n```python\n{func_source}\n```"

        self.documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "element_type": "function",
                    "element_name": f"{prefix}{node.name}",
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                },
            )
        )

        # Recursively visit child nodes (e.g., nested functions).
        self.generic_visit(node)


def clean_python_file(content: str, file_path: str) -> List[Document]:
    """
    Parses a Python file's content into an Abstract Syntax Tree (AST) and uses
    the `PythonCodeVisitor` to extract all classes and functions into a list of
    `Document` objects.

    This "deep cleaning" provides a structured, code-aware way to chunk
    Python files, keeping logical blocks of code together. This is highly beneficial
    for RAG systems as it ensures that a language model receives complete and
    contextually relevant code snippets.

    Args:
        content: The Python source code as a string.
        file_path: The path to the source file, used for metadata in the Documents.

    Returns:
        A list of `Document` objects, each representing a class or function.
        If parsing fails (e.g., due to `SyntaxError`), it returns a single `Document`
        containing the entire file content and an error message in its metadata.
    """
    # If the file is empty or contains only whitespace (e.g., __init__.py), return an
    # empty list.
    if not content.strip():
        return []

    try:
        # Step 1: Initialize our custom visitor.
        # This visitor will collect the Document objects as it traverses the AST.
        visitor = PythonCodeVisitor(content, file_path)
        # Step 2: Parse the source code into an AST.
        # `ast.parse` creates the tree structure from the Python code.
        tree = ast.parse(content)
        # Step 3: "Visit" the AST.
        # The `visitor.visit(tree)` call starts the traversal, which in turn triggers
        # `visit_ClassDef` and `visit_FunctionDef` methods for relevant nodes.
        visitor.visit(tree)
        # Step 3.1: Handle cases where we have config files ie if we have Python code
        # without Class or Function definitions.
        if not visitor.documents:
            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "element_type": "file",
                    },
                )
            ]
        # Step 4: Return the collected documents.
        # The `documents` list in the visitor now contains all the extracted classes and
        # functions.
        return visitor.documents
    except SyntaxError as e:
        return [
            Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "element_type": "file",  # Indicate that this Document represents the whole file
                    "parse_error": str(e),  # Include the parsing error for debugging
                },
            )
        ]


def clean_html_file(content: str) -> str:
    """
    Cleans HTML/Jinja files by converting them to Markdown.

    Args:
        content: The HTML/Jinja content to clean.

    Returns:
        Cleaned markdown content.
    """
    markdown_content = markdownify(content, strip=["script", "style"]).strip()

    # Fix malformed links like <{{ item.link }}>
    cleaned_content = re.sub(r"<(\{\{.*?\}\})>", r"\\1", markdown_content)
    # Fix escaped underscores
    cleaned_content = re.sub(r"\\_", "_", cleaned_content)

    return cleaned_content


# TODO: we might need a proper 3rd party JS parser
def clean_js_file(content: str) -> str:
    """
    Cleans JavaScript files by removing console statements.

    Args:
        content: The JavaScript content to clean.

    Returns:
        Cleaned JavaScript content.
    """
    content = re.sub(r"console\.\w+\(.*?\);?", "", content)
    content = re.sub(r"\n\s*\n", "\n", content)
    return content.strip()


# https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html
def clean_document(doc: Document) -> List[Document]:
    """
    Main dispatcher function to clean a document based on its file type.

    Args:
        doc: The Document object to clean.

    Returns:
        List of cleaned Document objects (may be multiple for Python files).
    """
    file_path = doc.metadata.get("source", "")
    file_suffix = Path(file_path).suffix

    content = doc.page_content

    if file_suffix == ".py":
        # Python files return multiple documents
        return clean_python_file(content, file_path)
    elif file_suffix in [".jinja", ".html"]:
        cleaned_content = clean_html_file(content)
    elif file_suffix == ".js":
        cleaned_content = clean_js_file(content)
    else:
        cleaned_content = content

    # For non-Python files, return single document
    return [Document(page_content=cleaned_content, metadata=doc.metadata)]
