from pathlib import Path

from langchain_community.document_loaders import GitLoader


# DOCS: https://python.langchain.com/docs/integrations/document_loaders/git/
def load_git_repository():
    """
    Loads files from a Git repository.

    The file_filter is used to select which files to load. It operates on the file
    paths of the cloned repository.
    """
    loader = GitLoader(
        clone_url="https://github.com/krstic-duci/zestify",
        repo_path="./src/tmp/",
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
    # NOTE: 50 documents ie files are loaded for above filter
    # print(f"Loaded {len(data)} documents.")

    # print("\n--- Verifying loaded files ---")
    # loaded_files = sorted(
    #     {doc.metadata.get("source").replace("tmp/", "") for doc in data}
    # )
    # for file_path in loaded_files:
    #     print(file_path)
    # print("--- Verification complete ---\\n")

    return data


# NOTE: naive approach as we wanna update specific files as they change
if __name__ == "__main__":
    if not Path("./src/tmp/").exists():
        load_git_repository()
    else:
        print(
            "Repository already cloned. To re-run and validate, "
            "please delete the './src/tmp/' directory and run this script again."
        )
