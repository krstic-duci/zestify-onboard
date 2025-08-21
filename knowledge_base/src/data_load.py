import logging
import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document

from .utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/krstic-duci/zestify"
REPO_PATH = "./src/tmp"


# DOCS: https://python.langchain.com/docs/integrations/document_loaders/git/
def load_git_repository(clean_existing: bool = False) -> List[Document]:
    """
    Loads files from a Git repository with error handling and idempotent cloning.

    Args:
        clean_existing: If True, remove existing repo directory before cloning.

    Returns:
        List of Document objects loaded from the repository.
    """
    repo_path = Path(REPO_PATH)

    if repo_path.exists():
        if clean_existing:
            logger.info("Removing existing repository at %s for fresh clone", repo_path)
            try:
                shutil.rmtree(repo_path)
            except Exception as e:
                logger.error("Failed to remove %s: %s", repo_path, e)
                return []
        else:
            logger.info("Repository exists at %s, reusing existing clone", repo_path)

    try:
        loader = GitLoader(
            clone_url=REPO_URL,
            repo_path=f"{REPO_PATH}/",
            file_filter=lambda file_path: file_path.endswith(
                (
                    ".py",
                    ".js",
                    "README.md",
                    ".jinja",
                    ".gitignore",
                    ".python-version",
                    "pyproject.toml",
                    "ruff.toml",
                    "uv.lock",
                )
            ),
        )
        data = loader.load()
        logger.info("Successfully loaded %d documents from %s", len(data), REPO_URL)
        return data
    except Exception as e:
        logger.error("Error loading repository %s: %s", REPO_URL, e, exc_info=True)
        return []


if __name__ == "__main__":
    repo_path = Path(REPO_PATH)
    if not repo_path.exists():
        load_git_repository(clean_existing=False)
    else:
        print(
            f"Repository already exists at {repo_path}. Use clean_existing=True to force fresh clone."
        )
