"""Shared agent context backed by a SWE-bench structure dict.

The structure dict (produced by ``create_structure`` in
``src/repo_handling/get_repo.py``) embeds the full text of every Python file in
the repository. The original checkout is discarded after it is built, so the
dict is the only artifact we carry through the pipeline.

Text navigation tools (read/ls/grep/glob) operate directly on this dict, so
they need no checkout. ``get_function_context`` uses jedi, which requires real
files on disk for import resolution, so it lazily materializes the dict to a
temporary directory the first time it is called.
"""

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Temp dirs created for jedi, cleaned up at process exit as a safety net.
_TEMP_ROOTS: set[str] = set()


def _index_structure(structure: dict) -> dict[str, str]:
    """Map every Python file to its source text, keyed by repo-relative path.

    Paths are formed by joining nested dict keys with '/', matching the
    convention used by the rest of the codebase (build_repo_skeleton,
    get_full_file_paths_and_classes_and_functions). Non-Python files carry no
    text in the structure dict and are therefore omitted.
    """
    file_texts: dict[str, str] = {}

    def walk(node: dict, prefix: str) -> None:
        for name, content in node.items():
            if not isinstance(content, dict):
                continue
            path = f"{prefix}/{name}" if prefix else name
            if "text" in content and "classes" in content:
                file_texts[path] = "\n".join(content.get("text", []))
            else:
                walk(content, path)

    walk(structure, "")
    return file_texts


def _materialize(file_texts: dict[str, str]) -> str:
    root = tempfile.mkdtemp(prefix="repoctx_")
    _TEMP_ROOTS.add(root)
    for rel_path, text in file_texts.items():
        dest = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Path(dest).write_text(text, encoding="utf-8")
    return root


@atexit.register
def _cleanup_all() -> None:
    for root in list(_TEMP_ROOTS):
        shutil.rmtree(root, ignore_errors=True)
    _TEMP_ROOTS.clear()


@dataclass
class RepoContext:
    """Run context shared by all repository-navigation tools.

    Construct one per SWE-bench instance from its structure dict and pass it as
    ``Runner.run(agent, prompt, context=RepoContext(structure=...))``.
    """

    structure: dict
    _file_texts: dict[str, str] | None = field(default=None, init=False, repr=False)
    _tmp_root: str | None = field(default=None, init=False, repr=False)

    @property
    def file_texts(self) -> dict[str, str]:
        if self._file_texts is None:
            self._file_texts = _index_structure(self.structure)
        return self._file_texts

    def materialize(self) -> str:
        """Write the Python files to a temp dir (once) and return its root."""
        if self._tmp_root is None:
            self._tmp_root = _materialize(self.file_texts)
        return self._tmp_root

    def cleanup(self) -> None:
        """Remove the materialized temp dir, if one was created."""
        if self._tmp_root is not None:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            _TEMP_ROOTS.discard(self._tmp_root)
            self._tmp_root = None
