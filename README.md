# Planning Module

A multi-agent system that turns a GitHub issue into a precise, **line-level change plan** for a software repository, without writing the code itself. It is built for [SWE-bench](https://www.swebench.com/) instances: given an issue and the repository at its base commit, the agents localize the bug, specify the fix, and emit a structured `FinalPlan` that a downstream code-writing step can execute.

The project ships **two pipelines**:

- **Blind pipeline** — never sees the solution. It reasons from the issue and the code to predict a plan. This is the system under test.
- **Ground-truth pipeline** — is handed the real merged diff (the gold patch) and works backwards from it to produce a reference plan, emitted in the _same_ `FinalPlan` format so the two can be compared entry-for-entry. This is the evaluation oracle.

Both pipelines are powered by the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) and the `gpt-5.4` model, with optional [LangSmith](https://www.langchain.com/langsmith) tracing.

---

## The pipelines, stage by stage

Every stage is an `Agent` (see `src/agents/definitions.py`) driven by a prompt in `src/agents/prompts/`. Stages 1–2 are pure LLM reasoning; stages 3–4 (and the ground-truth planner) navigate the live repository through five tools.

### Blind pipeline (`run_planning_pipeline`)

| #   | Agent                | Input                                              | Tools                                      | Output                                                                                |
| --- | -------------------- | -------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------- |
| 1   | `issue_preprocessor` | Raw issue text (`problem_statement`)               | none                                       | Structured JSON issue report (categorized, paraphrased)                               |
| 2   | `localization`       | Issue report + **repo skeleton** (signatures only) | none                                       | First-look localization report: ranked suspect files/functions, call paths to trace   |
| 3   | `specification`      | Issue report + localization report                 | read, ls, glob, grep, get_function_context | Detailed per-file implementation spec with `file:line` citations — **prose, no code** |
| 4   | `final_planner`      | Issue report + specification                       | read, ls, glob, grep, get_function_context | `FinalPlan` — files → hunks → line-level add/delete changes                           |

The blind pipeline never receives the gold patch. The skeleton (stage 2) lets localization reason over the whole repo cheaply before the deeper, tool-driven stages read real code.

### Ground-truth pipeline (`run_ground_truth_pipeline`)

| #   | Agent                  | Input                                               | Tools                                      | Output                                  |
| --- | ---------------------- | --------------------------------------------------- | ------------------------------------------ | --------------------------------------- |
| 1   | `issue_preprocessor`   | Raw issue text                                      | none                                       | Structured JSON issue report            |
| 2   | `ground_truth_planner` | Issue report + **gold patch** + `parse_patch` index | read, ls, glob, grep, get_function_context | `FinalPlan` — faithful to the gold diff |

The `ground_truth_planner` treats the gold patch as authoritative for _what_ changes and the issue report as the _why_. It re-anchors every line number against the **base-commit** repository (the patch's `+` numbers refer to the post-fix file and won't match), and writes each change as a natural-language instruction in the same format as the blind `final_planner` — so the two plans are directly comparable. The structured patch index comes from `parse_patch` in `src/repo_handling/get_patch_info.py`.

---

## Output format: `FinalPlan`

Both pipelines emit the same Pydantic model (`src/agents/states/structured_output.py`).

- `Change.type` is a literal: either `"add"` or `"delete"`.
- A **modification** is modeled as a `delete` of the old line(s) plus an `add` at the same location.
- `content` is the _perfect prompt_ for a single change: a self-contained natural-language instruction (no literal code) precise enough for a code-writing agent to implement that one change.
- `line` / `start_line` always refer to the **current (base-commit)** file.

Stored files are a JSON array of `{instance_id, plan}` records — identical structure for predicted and ground-truth runs:

```json
[
  {
    "instance_id": "psf__requests-1234",
    "plan": {
      "files": [
        {
          "file": "requests/models.py",
          "hunks": [
            {
              "start_line": 120,
              "changes": [
                {
                  "type": "delete",
                  "line": 122,
                  "content": "Remove the early return of None in PreparedRequest.prepare_body when data is a generator..."
                },
                {
                  "type": "add",
                  "line": 122,
                  "content": "Return self after streaming the generator body, setting Content-Length to ..."
                }
              ]
            }
          ]
        }
      ]
    }
  }
]
```

---

## Repository layout

```
src/
├── agents/
│   ├── definitions.py          # Agents + prompt builders; LangSmith setup; gpt-5.4
│   ├── orchestration.py        # Pipelines + batch storers + CLI entry point
│   ├── prompts/
│   │   ├── issue_preprocessor.txt
│   │   ├── localization.txt
│   │   ├── specification.txt
│   │   ├── final_planner.txt
│   │   └── ground_truth_planner.txt
│   ├── states/
│   │   └── structured_output.py  # FinalPlan / FilePlan / Hunk / Change
│   └── tools/
│       ├── repo_context.py       # RepoContext shared substrate
│       ├── navigation.py         # read / ls / glob_tool / grep
│       └── function_context.py   # get_function_context (jedi call graph)
├── repo_handling/
│   ├── get_repo.py             # clone @ base_commit → structure dict
│   └── get_patch_info.py       # parse_patch (unified diff → structured)
└── utils/
    ├── compress_file.py        # build_repo_skeleton / get_skeleton (libcst)
    ├── preprocess_data.py
    ├── parse_global_var.py
    └── utils.py                # jsonl/json IO helpers
results/
└── ground_truth_plans.json     # written by the ground-truth batch run
```

---

## Installation

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). Git must be available (repos are cloned at their base commit).

```bash
uv sync
```

Key dependencies: `openai-agents`, `datasets`, `jedi`, `libcst`, `pydantic`, `langsmith[openai-agents]`, `python-dotenv`.

## Configuration

Create a `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-...

# Optional — route agent traces to LangSmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=swe-bench-planning
```

Tracing is enabled only when `LANGSMITH_TRACING=true` **and** `LANGSMITH_API_KEY` is set; otherwise it is silently skipped (see `_setup_langsmith_tracing` in `definitions.py`).

---

## Usage

### CLI

Run the orchestrator as a module. By default it loads `princeton-nlp/SWE-bench_Lite` (`test` split).

```bash
# Blind pipeline over the first 5 instances → plans.json
uv run python -m src.agents.orchestration -n 5 -o my_plans.json

# Ground-truth pipeline → results/ground_truth_plans.json
uv run python -m src.agents.orchestration --ground-truth -n 5
```

| Flag             | Default                                          | Meaning                                                                             |
| ---------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `--dataset`      | `princeton-nlp/SWE-bench_Lite`                   | HuggingFace dataset name **or** a local `.jsonl` path                               |
| `--split`        | `test`                                           | Dataset split (HuggingFace only)                                                    |
| `-n`, `--limit`  | all                                              | Run only the first N instances                                                      |
| `-o`, `--output` | `plans.json` / `results/ground_truth_plans.json` | Combined JSON array output path                                                     |
| `--ground-truth` | off                                              | Build reference plans from each instance's gold patch instead of the blind pipeline |
| `--playground`   | `playground`                                     | Base dir for temporary clones                                                       |

Both batch runs rewrite the output file after each instance (crash-safe) and skip + log any instance that fails rather than aborting.
