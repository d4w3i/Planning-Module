import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import jedi
from agents import RunContextWrapper, function_tool

from src.agents.tools.repo_context import RepoContext


# ── Internal types ─────────────────────────────────────────────────────────────

FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]
FileCache = dict[str, tuple[str, ast.Module, list[str]]]


@dataclass(frozen=True)
class _FuncID:
    abs_path: str
    line: int


@dataclass
class _CalleeEntry:
    name: str
    rel_path: str
    def_line: int
    end_line: int
    snippet: str


@dataclass
class _CallerEntry:
    enclosing_name: str
    rel_path: str
    call_line: int
    def_line: int
    is_test: bool


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_test_file(abs_path: str) -> bool:
    p = Path(abs_path)
    name = p.name
    if name.startswith("test_") or name.endswith("_test.py") or name.endswith("_tests.py"):
        return True
    return any(part in ("test", "tests") for part in p.parts)


def _find_function_node(
    tree: ast.Module, function_name: str
) -> tuple[FuncNode | None, ast.ClassDef | None]:
    """Locate a function/method node. Accepts 'fn' or 'Class.fn' forms."""
    if "." in function_name:
        class_name, method_name = function_name.rsplit(".", 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                        return child, node
        return None, None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node, None
    return None, None


def _find_node_at_line(tree: ast.Module, line: int) -> FuncNode | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line:
            return node
    return None


def _extract_snippet(lines: list[str], fn_node: FuncNode) -> str:
    """Return signature lines + optional first docstring line + '...'"""
    if not fn_node.body:
        return lines[fn_node.lineno - 1] + "\n    ..."

    first_body = fn_node.body[0]
    sig_lines = lines[fn_node.lineno - 1 : first_body.lineno - 1]

    docstring_line = None
    if (
        isinstance(first_body, ast.Expr)
        and isinstance(first_body.value, ast.Constant)
        and isinstance(first_body.value.value, str)
    ):
        first_doc = first_body.value.value.strip().split("\n")[0]
        indent = " " * (fn_node.col_offset + 4)
        docstring_line = f'{indent}"""{first_doc}"""'

    body_indent = " " * (fn_node.col_offset + 4)
    parts = ["\n".join(sig_lines)]
    if docstring_line:
        parts.append(docstring_line)
    parts.append(f"{body_indent}...")
    return "\n".join(parts)


def _make_display_name(full_name: str | None, module_path: str) -> str:
    """Strip module path prefix from a jedi full_name for display."""
    if not full_name:
        return "<unknown>"
    stem = Path(module_path).stem
    parts = full_name.split(".")
    for i, part in enumerate(parts):
        if part == stem:
            remaining = ".".join(parts[i + 1 :])
            return remaining if remaining else full_name
    return ".".join(parts[-2:]) if len(parts) >= 2 else full_name


def _load_file(abs_path: str, file_cache: FileCache) -> tuple[str, ast.Module, list[str]] | None:
    if abs_path in file_cache:
        return file_cache[abs_path]
    try:
        source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        lines = source.splitlines()
        entry = (source, tree, lines)
        file_cache[abs_path] = entry
        return entry
    except Exception:
        return None


# ── Core analysis ──────────────────────────────────────────────────────────────


def _get_callees(
    fn_node: FuncNode,
    source: str,
    abs_path: str,
    script: jedi.Script,
    project: jedi.Project,
    repo_path: str,
    depth: int,
    visited: set[_FuncID],
    file_cache: FileCache,
) -> list[_CalleeEntry]:
    if depth == 0:
        return []

    seen: set[_FuncID] = set()
    result: list[_CalleeEntry] = []

    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            line, col = func.lineno, func.col_offset
        elif isinstance(func, ast.Attribute):
            line, col = func.end_lineno, func.end_col_offset - len(func.attr)
        else:
            continue

        try:
            definitions = script.goto(line, col, follow_imports=True)
        except Exception:
            continue

        for d in definitions:
            if not d.is_definition():
                continue
            if d.type not in ("function", "class"):
                continue
            if d.module_path is None or not str(d.module_path).endswith(".py"):
                continue
            # Skip stdlib and site-packages — only track calls within the repo
            if not str(d.module_path).startswith(repo_path):
                continue

            callee_id = _FuncID(str(d.module_path), d.line)
            if callee_id in visited or callee_id in seen:
                continue
            seen.add(callee_id)

            callee_abs = str(d.module_path)
            cached = _load_file(callee_abs, file_cache)
            if cached is None:
                continue
            callee_source, callee_tree, callee_lines = cached

            callee_fn = _find_node_at_line(callee_tree, d.line)
            if callee_fn is None:
                continue

            rel_path = os.path.relpath(callee_abs, repo_path)
            name = _make_display_name(d.full_name, callee_abs)
            snippet = _extract_snippet(callee_lines, callee_fn)

            result.append(_CalleeEntry(
                name=name,
                rel_path=rel_path,
                def_line=callee_fn.lineno,
                end_line=callee_fn.end_lineno,
                snippet=snippet,
            ))

            if depth > 1:
                callee_script = jedi.Script(code=callee_source, path=callee_abs, project=project)
                result.extend(_get_callees(
                    callee_fn, callee_source, callee_abs, callee_script,
                    project, repo_path, depth - 1,
                    visited | {callee_id} | seen, file_cache,
                ))

    return result


def _get_callers(
    fn_node: FuncNode,
    abs_path: str,
    script: jedi.Script,
    project: jedi.Project,
    repo_path: str,
    depth: int,
    visited: set[_FuncID],
    file_cache: FileCache,
) -> list[_CallerEntry]:
    if depth == 0:
        return []

    is_async = isinstance(fn_node, ast.AsyncFunctionDef)
    name_col = fn_node.col_offset + (10 if is_async else 4)

    try:
        refs = script.get_references(fn_node.lineno, name_col)
    except Exception:
        return []

    result: list[_CallerEntry] = []

    for r in refs:
        if r.is_definition() or r.module_path is None:
            continue

        try:
            parent = r.parent()
        except Exception:
            continue
        if parent is None:
            continue

        caller_abs = str(r.module_path)

        if parent.type == "module":
            enclosing_name = "<module>"
            caller_def_line = 1
        else:
            enclosing_name = _make_display_name(parent.full_name, caller_abs)
            caller_def_line = parent.line or r.line

        caller_id = _FuncID(caller_abs, caller_def_line)
        if caller_id in visited:
            continue

        result.append(_CallerEntry(
            enclosing_name=enclosing_name,
            rel_path=os.path.relpath(caller_abs, repo_path),
            call_line=r.line,
            def_line=caller_def_line,
            is_test=_is_test_file(caller_abs),
        ))

        if depth > 1 and parent.type != "module":
            cached = _load_file(caller_abs, file_cache)
            if cached is not None:
                caller_source, caller_tree, _ = cached
                caller_fn = _find_node_at_line(caller_tree, caller_def_line)
                if caller_fn is not None:
                    caller_script = jedi.Script(code=caller_source, path=caller_abs, project=project)
                    result.extend(_get_callers(
                        caller_fn, caller_abs, caller_script, project,
                        repo_path, depth - 1, visited | {caller_id}, file_cache,
                    ))

    return result


# ── Rendering ──────────────────────────────────────────────────────────────────


def _render_report(
    rel_path: str,
    fn_node: FuncNode,
    target_snippet: str,
    callees: list[_CalleeEntry],
    callers_visible: list[_CallerEntry],
    total_caller_count: int,
    test_caller_count: int,
    depth: int,
    include_tests: bool,
    max_callers: int,
) -> str:
    out = []

    out.append("## target function")
    out.append(f"**Location**: `{rel_path}:{fn_node.lineno}-{fn_node.end_lineno}`")
    out.append("```python")
    out.append(target_snippet)
    out.append("```")
    out.append("")

    out.append(f"## it calls (depth={depth})")
    if callees:
        for c in callees:
            out.append(f"- `{c.name}` defined at `{c.rel_path}:{c.def_line}`")
            out.append("```python")
            out.append(f"  {c.snippet}")
            out.append("```")
    else:
        out.append("- *(no resolvable calls found)*")
    out.append("")

    out.append(f"## called by (depth={depth})")
    if callers_visible:
        for c in callers_visible:
            test_tag = " *(test)*" if c.is_test else ""
            out.append(f"- `{c.enclosing_name}` at `{c.rel_path}:{c.call_line}`{test_tag}")
    else:
        out.append("- *(no callers found)*")
    out.append("")

    visible_count = total_caller_count - (test_caller_count if not include_tests else 0)
    was_truncated = len(callers_visible) < visible_count
    omitted_tests = not include_tests and test_caller_count > 0

    if was_truncated or omitted_tests:
        parts = [f"Showing {depth}-level depth. {total_caller_count} callers found"]
        if was_truncated:
            parts.append(f"showing first {max_callers}")
        if omitted_tests:
            parts.append("omitting test files unless include_tests=true")
        out.append("## truncation")
        out.append("- " + ", ".join(parts) + ".")

    return "\n".join(out)


# ── Tool entry point ───────────────────────────────────────────────────────────


@function_tool
def get_function_context(
    ctx: RunContextWrapper[RepoContext],
    file_path: str,
    function_name: str,
    depth: int = 1,
    include_tests: bool = False,
    max_callers: int = 10,
) -> str:
    """Produce a structured call-graph report for a Python function: its
    signature, what it calls (callees), and what calls it (callers), up to a
    given depth. Uses semantic analysis (jedi) to resolve imports and aliases.

    Args:
        file_path: Repo-relative path to the file (e.g. 'requests/adapters.py').
        function_name: Function name, optionally dot-qualified for methods
                       (e.g. 'acquire' or 'ConnectionPool.acquire').
        depth: Levels of transitive caller/callee depth to explore.
        include_tests: Whether to include callers from test files.
        max_callers: Maximum number of callers to show per level.
    """
    repo_path = ctx.context.materialize()
    abs_file_path = os.path.join(repo_path, file_path)

    if not os.path.isfile(abs_file_path):
        return f"Error: file not found: {file_path}"

    try:
        source = Path(abs_file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Error: could not parse {file_path}: {e}"

    fn_node, _ = _find_function_node(tree, function_name)
    if fn_node is None:
        return f"Error: function '{function_name}' not found in {file_path}"

    lines = source.splitlines()
    file_cache: FileCache = {abs_file_path: (source, tree, lines)}
    target_id = _FuncID(abs_file_path, fn_node.lineno)

    project = jedi.Project(path=repo_path)
    script = jedi.Script(code=source, path=abs_file_path, project=project)

    target_snippet = _extract_snippet(lines, fn_node)
    callees = _get_callees(
        fn_node, source, abs_file_path, script, project,
        repo_path, depth, {target_id}, file_cache,
    )
    all_callers = _get_callers(
        fn_node, abs_file_path, script, project,
        repo_path, depth, {target_id}, file_cache,
    )

    test_callers = [c for c in all_callers if c.is_test]
    non_test_callers = [c for c in all_callers if not c.is_test]
    callers_pool = all_callers if include_tests else non_test_callers
    callers_visible = callers_pool[:max_callers]

    rel_path = os.path.relpath(abs_file_path, repo_path)

    return _render_report(
        rel_path=rel_path,
        fn_node=fn_node,
        target_snippet=target_snippet,
        callees=callees,
        callers_visible=callers_visible,
        total_caller_count=len(all_callers),
        test_caller_count=len(test_callers),
        depth=depth,
        include_tests=include_tests,
        max_callers=max_callers,
    )
