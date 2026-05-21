import argparse
import asyncio
import json
import logging
import os

from agents import Runner

from src.agents.definitions import (
    final_planner,
    final_planner_prompt,
    ground_truth_planner,
    ground_truth_planner_prompt,
    issue_preprocessor,
    issue_preprocessor_prompt,
    localization,
    localization_prompt,
    specification,
    specification_prompt,
)
from src.agents.states.structured_output import FinalPlan
from src.agents.tools.repo_context import RepoContext
from src.repo_handling.get_repo import get_project_structure_from_scratch
from src.utils.compress_file import build_repo_skeleton
from src.utils.utils import load_jsonl

logger = logging.getLogger(__name__)


async def run_planning_pipeline(problem_statement: str, structure: dict) -> FinalPlan:
    """Run the full planning pipeline for one issue and return a FinalPlan.

    Stages:
      1. issue_preprocessor — raw issue text -> structured JSON issue report.
      2. localization       — issue report + repo skeleton -> first-look report.
      3. specification      — issue report + localization -> detailed change spec.
      4. final_planner      — issue report + specification -> structured FinalPlan.

    Stages 1-2 are pure reasoning. Stage 2 reasons over a skeleton built from the
    structure dict. Stages 3-4 navigate the repository through a shared
    RepoContext: the text tools read the structure dict directly, while
    get_function_context lazily materializes it to a temp dir for jedi. The
    context is reused across both stages so the repo is materialized at most once,
    and cleaned up when the pipeline finishes.

    Args:
        problem_statement: The raw GitHub issue text (SWE-bench problem_statement).
        structure: The repository structure dict (from create_structure).

    Returns:
        The FinalPlan produced by the final_planner agent.
    """
    # 1. Reorganize the raw issue into a structured report.
    logger.info("stage 1/4: issue_preprocessor")
    pre = await Runner.run(
        issue_preprocessor, issue_preprocessor_prompt(problem_statement)
    )
    issue_report = pre.final_output

    # 2. First-look localization from the repository skeleton.
    logger.info("stage 2/4: localization")
    skeleton = build_repo_skeleton(structure)
    loc = await Runner.run(localization, localization_prompt(issue_report, skeleton))
    localization_report = loc.final_output

    # Stages 3-4 navigate the live repository; share one context so the repo is
    # materialized for jedi at most once, then clean it up.
    ctx = RepoContext(structure=structure)
    try:
        # 3. Detailed, per-file implementation specification.
        logger.info("stage 3/4: specification")
        spec = await Runner.run(
            specification,
            specification_prompt(issue_report, localization_report),
            context=ctx,
        )
        specification_doc = spec.final_output

        # 4. Structured, line-level change plan.
        logger.info("stage 4/4: final_planner")
        plan = await Runner.run(
            final_planner,
            final_planner_prompt(issue_report, specification_doc),
            context=ctx,
        )
    finally:
        ctx.cleanup()

    return plan.final_output


async def run_ground_truth_pipeline(
    problem_statement: str, structure: dict, diff_patch: str
) -> FinalPlan:
    """Produce the GROUND-TRUTH FinalPlan for one issue from its gold patch.

    Unlike run_planning_pipeline, which never sees the solution, this pipeline is
    handed the real merged diff and works backwards from it:
      1. issue_preprocessor   — raw issue text -> structured JSON issue report.
      2. ground_truth_planner — issue report + gold patch -> structured FinalPlan,
         with line numbers verified against the base-commit repository via tools.

    The plan is emitted in the same FinalPlan format as the blind pipeline, so it
    can serve as the reference for evaluating blind plans.

    Args:
        problem_statement: The raw GitHub issue text (SWE-bench problem_statement).
        structure: The repository structure dict (base commit, from create_structure).
        diff_patch: The gold solution patch (SWE-bench 'patch' field).

    Returns:
        The ground-truth FinalPlan produced by the ground_truth_planner agent.
    """
    logger.info("ground-truth stage 1/2: issue_preprocessor")
    pre = await Runner.run(
        issue_preprocessor, issue_preprocessor_prompt(problem_statement)
    )
    issue_report = pre.final_output

    # The planner navigates the base-commit repo to anchor line numbers.
    ctx = RepoContext(structure=structure)
    try:
        logger.info("ground-truth stage 2/2: ground_truth_planner")
        plan = await Runner.run(
            ground_truth_planner,
            ground_truth_planner_prompt(issue_report, diff_patch),
            context=ctx,
        )
    finally:
        ctx.cleanup()

    return plan.final_output


async def run_ground_truth_from_instance(
    instance: dict, playground: str = "playground"
) -> FinalPlan:
    """Build the base-commit structure for a SWE-bench instance and produce its
    ground-truth FinalPlan from the instance's gold patch.

    Args:
        instance: A SWE-bench instance carrying 'repo', 'base_commit',
            'instance_id', 'problem_statement', and 'patch' (the gold diff).
        playground: Base directory for the temporary clone (created if absent).

    Returns:
        The ground-truth FinalPlan for the instance.
    """
    d = await asyncio.to_thread(
        get_project_structure_from_scratch,
        instance["repo"],
        instance["base_commit"],
        instance["instance_id"],
        playground,
    )
    return await run_ground_truth_pipeline(
        instance["problem_statement"], d["structure"], instance["patch"]
    )


async def run_from_instance(instance: dict, playground: str = "playground") -> FinalPlan:
    """Build the repository structure for a SWE-bench instance and plan it.

    Clones the repo at the instance's base commit, builds the structure dict,
    discards the clone, then runs the planning pipeline. The repo must be one of
    the supported repos in get_repo.repo_to_top_folder.

    Args:
        instance: A SWE-bench instance carrying 'repo', 'base_commit',
            'instance_id', and 'problem_statement'.
        playground: Base directory for the temporary clone (created if absent).

    Returns:
        The FinalPlan for the instance.
    """
    d = await asyncio.to_thread(
        get_project_structure_from_scratch,
        instance["repo"],
        instance["base_commit"],
        instance["instance_id"],
        playground,
    )
    return await run_planning_pipeline(instance["problem_statement"], d["structure"])


async def run_and_store(
    instances: list[dict],
    output_path: str = "plans.json",
    playground: str = "playground",
) -> list[dict]:
    """Plan a batch of SWE-bench instances and store every FinalPlan in one
    JSON array file.

    Each entry is ``{"instance_id": ..., "plan": <FinalPlan dict>}``. The file is
    rewritten after each instance, so completed plans survive a crash mid-batch.
    An instance that fails to plan is logged and skipped rather than aborting the
    whole batch.

    Args:
        instances: SWE-bench instances (each with 'repo', 'base_commit',
            'instance_id', 'problem_statement').
        output_path: Path to the combined JSON array file.
        playground: Base directory for the temporary clones.

    Returns:
        The list of stored {instance_id, plan} records.
    """
    results: list[dict] = []
    for instance in instances:
        instance_id = instance.get("instance_id", "<unknown>")
        try:
            plan = await run_from_instance(instance, playground=playground)
        except Exception:
            logger.exception("failed to plan %s", instance_id)
            continue
        results.append({"instance_id": instance_id, "plan": plan.model_dump()})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("stored plan for %s (%d total)", instance_id, len(results))
    return results


async def run_and_store_ground_truth(
    instances: list[dict],
    output_path: str = "results/ground_truth_plans.json",
    playground: str = "playground",
) -> list[dict]:
    """Build ground-truth plans for a batch of SWE-bench instances and store
    every FinalPlan in one JSON array file.

    The output mirrors run_and_store's format exactly — a list of
    ``{"instance_id": ..., "plan": <FinalPlan dict>}`` — so a ground-truth file
    can be compared entry-for-entry against a blind-pipeline file. The output's
    parent directory is created if missing. The file is rewritten after each
    instance, so completed plans survive a crash mid-batch. An instance that
    fails to plan is logged and skipped rather than aborting the whole batch.

    Args:
        instances: SWE-bench instances (each with 'repo', 'base_commit',
            'instance_id', 'problem_statement', and 'patch').
        output_path: Path to the combined JSON array file.
        playground: Base directory for the temporary clones.

    Returns:
        The list of stored {instance_id, plan} records.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results: list[dict] = []
    for instance in instances:
        instance_id = instance.get("instance_id", "<unknown>")
        try:
            plan = await run_ground_truth_from_instance(instance, playground=playground)
        except Exception:
            logger.exception("failed to build ground-truth plan for %s", instance_id)
            continue
        results.append({"instance_id": instance_id, "plan": plan.model_dump()})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(
            "stored ground-truth plan for %s (%d total)", instance_id, len(results)
        )
    return results


def load_instances(dataset: str, split: str, limit: int | None) -> list[dict]:
    """Load SWE-bench instances from a HuggingFace dataset or a local JSONL file.

    Args:
        dataset: HuggingFace dataset name (e.g. 'princeton-nlp/SWE-bench_Lite')
            or a path to a local .jsonl file.
        split: Dataset split to load (only used for HuggingFace datasets).
        limit: If set, keep only the first N instances.

    Returns:
        A list of instance dicts.
    """
    if dataset.endswith(".jsonl") or os.path.exists(dataset):
        rows = load_jsonl(dataset)
    else:
        from datasets import load_dataset

        rows = [dict(row) for row in load_dataset(dataset, split=split)]

    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the planning pipeline over SWE-bench Lite (or a subset) "
        "and store every FinalPlan in a single JSON array file."
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name or a local .jsonl path "
        "(default: princeton-nlp/SWE-bench_Lite).",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split (default: test)."
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Run only the first N instances (default: the whole dataset).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to the combined JSON array output (default: plans.json for the "
        "blind pipeline, results/ground_truth_plans.json with --ground-truth).",
    )
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Build ground-truth plans from each instance's gold patch instead of "
        "running the blind planning pipeline.",
    )
    parser.add_argument(
        "--playground",
        default="playground",
        help="Base directory for temporary repo clones (default: playground).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    instances = load_instances(args.dataset, args.split, args.limit)
    logger.info("loaded %d instances from %s", len(instances), args.dataset)

    if args.ground_truth:
        output = args.output or "results/ground_truth_plans.json"
        store = run_and_store_ground_truth
    else:
        output = args.output or "plans.json"
        store = run_and_store

    results = asyncio.run(
        store(instances, output_path=output, playground=args.playground)
    )
    logger.info("done: %d/%d plans stored to %s", len(results), len(instances), output)


if __name__ == "__main__":
    main()
