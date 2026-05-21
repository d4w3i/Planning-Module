import json
import logging
import os
from pathlib import Path

from agents import Agent, ToolSearchTool, set_default_openai_client
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agents.states.structured_output import FinalPlan
from src.agents.tools.function_context import get_function_context
from src.agents.tools.navigation import glob_tool, grep, ls, read
from src.agents.tools.repo_context import RepoContext
from src.repo_handling.get_patch_info import parse_patch


def _setup_langsmith_tracing() -> None:
    """Route the agents SDK's traces to LangSmith when LANGSMITH_TRACING=true."""
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return
    if not os.environ.get("LANGSMITH_API_KEY"):
        logging.getLogger(__name__).warning(
            "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is unset; tracing disabled."
        )
        return
    from agents import set_trace_processors
    from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

    set_trace_processors([OpenAIAgentsTracingProcessor()])


# Load .env, register the client all agents run against, and enable tracing.
load_dotenv()
set_default_openai_client(AsyncOpenAI())
_setup_langsmith_tracing()

MODEL = "gpt-5.4"

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


issue_preprocessor = Agent(
    name="issue_preprocessor",
    instructions=_load_prompt("issue_preprocessor.txt"),
    model=MODEL,
)


def issue_preprocessor_prompt(problem_statement: str) -> str:
    return f"<issue>\n{problem_statement}\n</issue>"


localization = Agent(
    name="localization",
    instructions=_load_prompt("localization.txt"),
    model=MODEL,
)


def localization_prompt(issue_report: str, repo_skeleton: str) -> str:
    return (
        f"<issue_report>\n{issue_report}\n</issue_report>\n\n"
        f"<repo_skeleton>\n{repo_skeleton}\n</repo_skeleton>"
    )


specification = Agent[RepoContext](
    name="specification",
    instructions=_load_prompt("specification.txt"),
    tools=[read, ls, glob_tool, grep, get_function_context],
    model=MODEL,
)


def specification_prompt(issue_report: str, localization_report: str) -> str:
    return (
        f"<issue_report>\n{issue_report}\n</issue_report>\n\n"
        f"<localization_report>\n{localization_report}\n</localization_report>"
    )


final_planner = Agent[RepoContext](
    name="final_planner",
    instructions=_load_prompt("final_planner.txt"),
    tools=[read, ls, glob_tool, grep, get_function_context],
    output_type=FinalPlan,
    model=MODEL,
)


def final_planner_prompt(issue_report: str, specification: str) -> str:
    return (
        f"<issue_report>\n{issue_report}\n</issue_report>\n\n"
        f"<specification>\n{specification}\n</specification>"
    )


ground_truth_planner = Agent[RepoContext](
    name="ground_truth_planner",
    instructions=_load_prompt("ground_truth_planner.txt"),
    tools=[read, ls, glob_tool, grep, get_function_context],
    output_type=FinalPlan,
    model=MODEL,
)


def ground_truth_planner_prompt(issue_report: str, diff_patch: str) -> str:
    parsed = parse_patch(diff_patch)
    return (
        f"<issue_report>\n{issue_report}\n</issue_report>\n\n"
        f"<gold_patch>\n{diff_patch}\n</gold_patch>\n\n"
        f"<gold_patch_parsed>\n{json.dumps(parsed, indent=2)}\n</gold_patch_parsed>"
    )
