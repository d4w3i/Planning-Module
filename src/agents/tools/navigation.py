import fnmatch
import re

from agents import RunContextWrapper, function_tool

from src.agents.tools.repo_context import RepoContext


def _norm(path: str) -> str:
    """Normalize a repo-relative path: drop leading './' and surrounding '/'."""
    p = path.strip()
    if p in (".", "./", "/", ""):
        return ""
    if p.startswith("./"):
        p = p[2:]
    return p.strip("/")


# ── Read ──────────────────────────────────────────────────────────────────────


@function_tool
def read(
    ctx: RunContextWrapper[RepoContext],
    file_path: str,
    offset: int = 1,
    limit: int = 2000,
) -> str:
    """Read a Python file from the repository, returning content with line
    numbers (cat -n format).

    Args:
        file_path: Repo-relative path to the file (e.g. 'requests/adapters.py').
        offset: Starting line number (1-indexed).
        limit: Maximum number of lines to return.
    """
    rel = _norm(file_path)
    texts = ctx.context.file_texts
    if rel not in texts:
        if any(p.startswith(rel + "/") for p in texts):
            return f"Error: {file_path} is a directory, not a file"
        return f"Error: file not found: {file_path}"

    lines = texts[rel].splitlines()
    start = max(0, offset - 1)
    chunk = lines[start : start + limit]
    return "\n".join(f"{start + i + 1:6}\t{line}" for i, line in enumerate(chunk))


# ── LS ────────────────────────────────────────────────────────────────────────


@function_tool
def ls(ctx: RunContextWrapper[RepoContext], path: str = ".") -> str:
    """List the contents of a directory in the repository, one entry per line.
    Directories are marked with a trailing '/'. Only Python files are tracked.

    Args:
        path: Repo-relative directory path. Defaults to the repository root.
    """
    base = _norm(path)
    prefix = base + "/" if base else ""
    dirs: set[str] = set()
    files: set[str] = set()

    for p in ctx.context.file_texts:
        if prefix and not p.startswith(prefix):
            continue
        rest = p[len(prefix) :]
        if "/" in rest:
            dirs.add(rest.split("/", 1)[0] + "/")
        elif rest:
            files.add(rest)

    if not dirs and not files:
        return f"Error: path not found: {path}" if base else "(empty repository)"
    return "\n".join(sorted(dirs) + sorted(files))


# ── Glob ──────────────────────────────────────────────────────────────────────


@function_tool
def glob_tool(
    ctx: RunContextWrapper[RepoContext],
    pattern: str,
    base_dir: str = ".",
) -> str:
    """Find Python files matching a glob pattern, relative to base_dir.
    Returns one repo-relative path per line, sorted.

    Args:
        pattern: Glob pattern, e.g. '**/*.py' or 'src/*.py'.
        base_dir: Repo-relative directory to search from. Defaults to repo root.
    """
    base = _norm(base_dir)
    prefix = base + "/" if base else ""
    matches = []
    for p in sorted(ctx.context.file_texts):
        if prefix and not p.startswith(prefix):
            continue
        rel = p[len(prefix) :]
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p, pattern):
            matches.append(p)
    if not matches:
        return f"No files matched: {pattern}"
    return "\n".join(matches)


# ── Grep ──────────────────────────────────────────────────────────────────────


@function_tool
def grep(
    ctx: RunContextWrapper[RepoContext],
    pattern: str,
    path: str = ".",
    glob_pattern: str = "**/*",
    ignore_case: bool = False,
    max_results: int = 100,
) -> str:
    """Search for a regex pattern across Python file contents, similar to grep -r.
    Returns matching lines in 'file:line_num:content' format.

    Args:
        pattern: Regex pattern to search for.
        path: Repo-relative directory (or file) to search in.
        glob_pattern: Filter files by glob, e.g. '**/*.py'. Default searches all.
        ignore_case: Case-insensitive matching.
        max_results: Maximum number of matches to return.
    """
    base = _norm(path)
    prefix = base + "/" if base else ""
    flags = re.IGNORECASE if ignore_case else 0

    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex: {e}"

    texts = ctx.context.file_texts
    filter_glob = glob_pattern not in ("**/*", "*", "")
    results, count = [], 0

    for fp in sorted(texts):
        if base and not (fp == base or fp.startswith(prefix)):
            continue
        rel = fp[len(prefix) :] if prefix else fp
        if filter_glob and not (
            fnmatch.fnmatch(rel, glob_pattern) or fnmatch.fnmatch(fp, glob_pattern)
        ):
            continue
        for i, line in enumerate(texts[fp].splitlines(), 1):
            if regex.search(line):
                results.append(f"{fp}:{i}:{line}")
                count += 1
                if count >= max_results:
                    results.append(f"[truncated at {max_results} results]")
                    return "\n".join(results)

    return "\n".join(results) if results else "No matches found."
