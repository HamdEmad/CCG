#!/usr/bin/env python3
"""
CLI entry point for the component lookup pipeline.

Usage:
    # Run on a message passed inline
    python scripts/run_pipeline.py --message "need lifecycle for FVD16H0474M22 from KYOCERA AVX"

    # Run on a message stored in a file
    python scripts/run_pipeline.py --message-file customer_message.txt

    # Run on a message stored in a JSON file
    python scripts/run_pipeline.py --message-json customer_message.json

    # Keep the workspace on disk after the run (for inspection / resume)
    PIPELINE_WORKSPACE_PERSISTENT=true python scripts/run_pipeline.py ...

Exit codes:
    0  - pipeline ran and every part completed successfully
    1  - pipeline ran but one or more parts failed (partial result; see output)
    2  - pipeline could not start at all (bad config, bad input, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# Allow running this script directly without installing the package.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pipeline.config import get_settings          # noqa: E402
from pipeline.state import PipelineResult         # noqa: E402
from pipeline.state_io import init_state          # noqa: E402
from pipeline.orchestrator import run_pipeline    # noqa: E402
from pipeline.state import PartAttributeResult, PartDetails, PartStatus, ResultSource  # noqa: E402
from integrations.llm_client import get_token_usage, reset_token_usage  # noqa: E402

logger = logging.getLogger("component_lookup")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the component lookup pipeline on a customer message.",
    )

    message_group = parser.add_mutually_exclusive_group(required=True)
    message_group.add_argument(
        "--message", type=str,
        help="Customer message text, passed directly on the command line.",
    )
    message_group.add_argument(
        "--message-file", type=Path,
        help="Path to a text file containing the customer message.",
    )
    message_group.add_argument(
        "--message-json", type=Path,
        help="Path to a JSON file containing the customer message.",
    )

    parser.add_argument(
        "--output", type=Path, default=Path("result.json"),
        help="Write the final result as JSON to this file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Message loading
# ---------------------------------------------------------------------------

def _load_customer_message(args: argparse.Namespace) -> str:
    if args.message is not None:
        return args.message

    if args.message_file is not None:
        if not args.message_file.exists():
            raise FileNotFoundError(f"Message file does not exist: {args.message_file}")
        text = args.message_file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Message file is empty: {args.message_file}")
        return text

    if args.message_json is not None:
        if not args.message_json.exists():
            raise FileNotFoundError(f"Message JSON file does not exist: {args.message_json}")
        raw_json = args.message_json.read_text(encoding="utf-8").strip()
        if not raw_json:
            raise ValueError(f"Message JSON file is empty: {args.message_json}")
        data = json.loads(raw_json)
        text = None
        if isinstance(data, dict):
            text = data.get("message") or data.get("customer_message") or data.get("text")
            if not text and data:
                text = next(iter(data.values()))
        elif isinstance(data, str):
            text = data
        elif isinstance(data, list) and data:
            text = data[1]
        if not text or not str(text).strip():
            raise ValueError(f"No valid message content found in JSON: {args.message_json}")
        return str(text).strip()

    raise ValueError("No customer message source specified.")


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def _print_summary(result: PipelineResult) -> None:
    total = result.total_parts
    print()
    print(f"Pipeline finished: {len(result.completed)}/{total} completed, "
          f"{len(result.failed)}/{total} failed.")
    print()

    if result.completed:
        print("Completed:")
        for r in result.completed:
            print(f"  - {r.part.part} ({r.part.manufacturer})")
    if result.failed:
        print()
        print("Failed:")
        for r in result.failed:
            print(f"  - {r.part.part} ({r.part.manufacturer}): {r.error}")

    tokens = get_token_usage()
    if tokens["total_tokens"] > 0:
        print()
        print("Token Usage:")
        print(f"  Input:  {tokens['input_tokens']:,}")
        print(f"  Output: {tokens['output_tokens']:,}")
        print(f"  Total:  {tokens['total_tokens']:,}")


def _build_pipeline_result(
    customer_message: str,
    completed_dicts: list[dict],
    failed_dicts: list[dict],
) -> PipelineResult:
    """Convert raw result dicts from the orchestrator into PipelineResult."""

    def _to_attr_result(d: dict, is_failed: bool) -> PartAttributeResult:
        pd = d.get("part_details", {})
        part = PartDetails.model_validate(pd)
        return PartAttributeResult(
            part=part,
            attributes=d.get("attributes", {}),
            landing_page=d.get("landing_page"),
            source=ResultSource(d["source"]) if d.get("source") else None,
            status=PartStatus.FAILED if is_failed else PartStatus.EXTRACTED,
            confidence=d.get("confidence", 0.0),
            found_on=d.get("found_on"),
            match_type=d.get("match_type"),
            error=d.get("error"),
        )

    completed = [_to_attr_result(d, False) for d in completed_dicts]
    failed    = [_to_attr_result(d, True)  for d in failed_dicts]

    any_requested = any(
        bool(r.part.attributes) or bool(r.part.crosses) for r in completed
    )
    if not completed:
        resolution = None
    elif not any_requested:
        noun = "parts" if len(completed) > 1 else "part"
        resolution = f"The {noun} have been successfully added."
    else:
        noun = "parts" if len(completed) > 1 else "part"
        resolution = (
            f"We have successfully added the {noun} and are currently "
            f"working on adding the required attributes."
        )

    return PipelineResult(
        customer_request=customer_message,
        completed=completed,
        failed=failed,
        resolution=resolution,
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> PipelineResult:
    customer_message = _load_customer_message(args)
    logger.info("Customer message loaded.")

    settings = get_settings()
    settings.require_llm_api_key()

    message_id = f"run-{uuid.uuid4().hex[:12]}"
    reset_token_usage()

    persistent = settings.pipeline_workspace_persistent
    if persistent:
        workspace = Path(settings.pipeline_workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        logger.info("Using persistent workspace: %s", workspace)
        cleanup = False
    else:
        workspace = Path(tempfile.mkdtemp(prefix="ccg_pipeline_"))
        logger.info("Using temporary workspace: %s", workspace)
        cleanup = True

    try:
        init_state(message_id, customer_message, workspace)
        completed_dicts, failed_dicts = run_pipeline(workspace)
        return _build_pipeline_result(customer_message, completed_dicts, failed_dicts)
    finally:
        if cleanup:
            shutil.rmtree(workspace, ignore_errors=True)
            logger.debug("Temporary workspace cleaned up.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    try:
        result = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(str(e))
        return 2
    except Exception:
        logger.exception("Pipeline run failed unexpectedly.")
        return 2

    _print_summary(result)

    if args.output:
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Full result written to: %s", args.output)

    return 0 if not result.failed else 1


if __name__ == "__main__":
    sys.exit(main())