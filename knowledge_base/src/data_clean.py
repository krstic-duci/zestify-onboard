import ast
import re
import textwrap
from pathlib import Path

from langchain_core.documents import Document
from markdownify import markdownify  # type: ignore # ignore


class PythonCodeVisitor(ast.NodeVisitor):
    """
    An AST visitor to extract structured information from Python code.
    """

    def __init__(self, source_code):
        self.source_lines = source_code.splitlines(keepends=True)
        self.results = []
        self.current_class = None

    def _get_source_segment(self, node):
        """A helper function to extract the full source code of an AST node."""
        start_line, start_col = node.lineno - 1, node.col_offset
        end_line, end_col = node.end_lineno - 1, node.end_col_offset

        if start_line == end_line:
            return self.source_lines[start_line][start_col:end_col]

        first_line = self.source_lines[start_line][start_col:]
        inner_lines = self.source_lines[start_line + 1 : end_line]
        last_line = self.source_lines[end_line][:end_col]

        return "".join([first_line] + inner_lines + [last_line])

    def visit_Import(self, node):
        for alias in node.names:
            self.results.append(f"Imports: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.results.append(f"Imports: from {node.module} import {alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        parent_class = self.current_class
        self.current_class = node.name
        docstring = ast.get_docstring(node)
        class_source = self._get_source_segment(node)
        self.results.append(f"\n--- Class: {node.name} ---")
        if docstring:
            self.results.append(f"Docstring:\n{textwrap.dedent(docstring)}")
        self.results.append(f"Source Code:\n```python\n{class_source}\n```")
        self.generic_visit(node)
        self.current_class = parent_class

    def visit_FunctionDef(self, node):
        if node.name.startswith("_") and not self.current_class:
            self.generic_visit(node)
            return
        func_source = self._get_source_segment(node)
        docstring = ast.get_docstring(node)
        prefix = f"{self.current_class}." if self.current_class else ""
        self.results.append(f"\n--- Function: {prefix}{node.name} ---")
        if docstring:
            self.results.append(f"Docstring:\n{textwrap.dedent(docstring)}")
        self.results.append(f"Source Code:\n```python\n{func_source}\n```")
        self.generic_visit(node)


def clean_python_file(content: str) -> str:
    """
    Deep cleans a Python file by parsing it into an AST and extracting
    structured information about its contents.
    """
    if not content.strip():
        return ""
    try:
        visitor = PythonCodeVisitor(content)
        tree = ast.parse(content)
        visitor.visit(tree)
        return "\n".join(visitor.results)
    except SyntaxError:
        return content


def clean_html_file(content: str) -> str:
    """
    Cleans HTML/Jinja files by converting them to Markdown, and then
    post-processing the output to fix artifacts from the conversion.
    """
    # Convert HTML to Markdown
    markdown_content = markdownify(content, strip=["script", "style"]).strip()

    # Fix malformed links like <{{ item.link }}>
    cleaned_content = re.sub(r"<(\{\{.*?\}\})>", r"\\1", markdown_content)
    # Fix escaped underscores
    cleaned_content = re.sub(r"\\_", "_", cleaned_content)

    return cleaned_content


def clean_js_file(content: str) -> str:
    """
    Cleans JavaScript files by removing all console statements, preserving comments.
    """
    content = re.sub(r"console\.\w+\(.*?\);?", "", content)
    content = re.sub(r"\n\s*\n", "\n", content)
    return content.strip()


def clean_document(doc: Document) -> Document:
    """
    Main dispatcher function to clean a document based on its file type.
    """
    file_path = doc.metadata.get("source", "")
    file_suffix = Path(file_path).suffix

    content = doc.page_content
    if file_suffix == ".py":
        cleaned_content = clean_python_file(content)
    elif file_suffix in [".jinja", ".html"]:
        cleaned_content = clean_html_file(content)
    elif file_suffix == ".js":
        cleaned_content = clean_js_file(content)
    else:
        cleaned_content = content

    return Document(page_content=cleaned_content, metadata=doc.metadata)
