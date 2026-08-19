#!/usr/bin/env python3
"""
Batch runner for the stage-first pipeline.

Reads a JSON file of {key: customer_message} pairs, creates one state
file per message, and runs all pipeline stages across all messages at
once. The workspace is persistent so runs can be resumed after a crash.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running directly without installing the package.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pipeline.config import get_settings          # noqa: E402
from pipeline.state_io import init_state          # noqa: E402
from pipeline.orchestrator import run_pipeline    # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_runner")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run batch pipeline.")
    parser.add_argument("input_json", nargs="?", default="random_subset.json", help="Path to input JSON file.")
    args = parser.parse_args()

    json_path = Path(args.input_json)
    if not json_path.exists():
        fallback = Path("src") / args.input_json
        if fallback.exists():
            json_path = fallback
        else:
            logger.error("Input JSON file not found: %s", args.input_json)
            sys.exit(1)

    out_dir   = Path("batch_results")
    out_dir.mkdir(exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    settings = get_settings()
    workspace = Path(settings.pipeline_workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    logger.info("Workspace: %s", workspace)

    queued = 0
    active_message_ids = []
    for key, message in data.items():
        out_file = out_dir / f"{key}.json"
        if out_file.exists():
            # Already fully completed and written to output directory
            logger.info("Skipping %s — completed output file already exists.", key)
            continue
            
        message_id = f"batch-{key}"
        active_message_ids.append(message_id)
        
        json_path_state = workspace / f"{message_id}.json"
        if json_path_state.exists():
            logger.info("Queued existing state for %s", key)
            continue
            
        init_state(message_id, message, workspace)
        logger.info("Queued new: %s", key)
        queued += 1

    if not active_message_ids:
        logger.info("All messages are already completed in batch_results. Nothing to do.")
        return

    logger.info("Running pipeline across %d active message(s) (%d new)...", len(active_message_ids), queued)

    completed, failed = run_pipeline(workspace, message_ids=active_message_ids)

    # Write per-key result files for backward compatibility
    from pipeline.state_io import load_state
    for key in data:
        out_file = out_dir / f"{key}.json"
        
        # Load the specific state file for this key
        state_path = workspace / f"batch-{key}.json"
        if not state_path.exists():
            continue

        try:
            state = load_state(state_path)
        except Exception as e:
            logger.warning("Could not load state file %s for key %s: %s", state_path.name, key, e)
            continue

        combined_completed = []
        combined_failed = []
        for part in state.get("parts", []):
            final_status = part.get("final_status", "pending")
            final_result = part.get("final_result") or {}
            res = {
                "part_details": part.get("part_details", {}),
                **final_result,
            }
            if final_status == "completed":
                combined_completed.append(res)
            elif final_status == "failed":
                combined_failed.append(res)

        combined = {
            "completed": combined_completed,
            "failed":    combined_failed,
        }
        out_file.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")

    logger.info("Done writing batch results.")


if __name__ == "__main__":
    main()
