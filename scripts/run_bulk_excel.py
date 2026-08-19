"""
Bulk Excel runner for the stage-first pipeline.

Reads part requests from an Excel sheet, creates one JSON state file per
row in the workspace, then runs all pipeline stages across all rows
together. The workspace is persistent (PIPELINE_WORKSPACE_PERSISTENT
behaviour is always on for bulk runs), so a crash can be resumed without
reprocessing already-completed rows.

Resume behaviour:
    If a JSON file already exists for a given row, it is left untouched.
    The orchestrator will skip any stage whose status is already "done".
    Re-running the script after a crash automatically resumes from the
    exact failure point.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running directly without installing the package.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

load_dotenv(project_root / ".env", override=True)

from pipeline.config import get_settings              # noqa: E402
from pipeline.state_io import init_state              # noqa: E402
from pipeline.orchestrator import run_pipeline        # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bulk_excel")

INPUT_EXCEL = project_root / "LC and ROHS DC_Hamdi.xlsx"
OUTPUT_CSV  = project_root / "bulk_extracted_results.csv"

# Set to True to stop after N rows (useful for testing).
LIMIT_RUN       = True
LIMIT_RUN_COUNT = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_customer_request(se_pn: str, se_man: str, flag: str) -> str:
    return (
        f"Hi, please retrieve the {flag} details for part {se_pn} "
        f"manufactured by {se_man}.\n\n"
        f"Manufacturer part:\n{se_pn}\n\n"
        f"Manufacturer:\n{se_man}\n"
    )


def _write_csv_results(
    df: pd.DataFrame,
    completed: list[dict],
    failed: list[dict],
    output_csv: Path,
) -> None:
    """Write or append pipeline results to the output CSV."""
    fieldnames = list(df.columns) + [
        "status", "source", "confidence", "landing_page", "extracted_attributes",
    ]
    file_exists = output_csv.exists()

    # Build a lookup from part number → result for fast access
    result_lookup: dict[str, dict] = {}
    for r in completed:
        pn = r.get("part_details", {}).get("part", "")
        result_lookup[pn] = {**r, "_status": "completed"}
    for r in failed:
        pn = r.get("part_details", {}).get("part", "")
        result_lookup[pn] = {**r, "_status": "failed"}

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for _, row in df.iterrows():
            se_pn = str(row["SE_PN"]).strip()
            result = result_lookup.get(se_pn)
            if result is None:
                continue

            final_status = result.get("_status", "failed")
            out_row = dict(row)
            out_row["status"]               = final_status
            out_row["source"]               = result.get("source", "None")
            out_row["confidence"]           = result.get("confidence", 0.0)
            out_row["landing_page"]         = result.get("landing_page", "")
            out_row["extracted_attributes"] = json.dumps(
                result.get("attributes", result.get("error", {})), default=str
            )
            writer.writerow(out_row)
        f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bulk() -> None:
    settings = get_settings()
    settings.require_llm_api_key()

    # Workspace is always persistent for bulk runs
    workspace = project_root / settings.pipeline_workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)
    logger.info("Workspace: %s", workspace)

    df = pd.read_excel(INPUT_EXCEL)
    total_rows = len(df)
    queued = 0

    for idx, row in df.iterrows():
        se_pn  = str(row["SE_PN"]).strip()
        se_man = str(row["SE_MAN"]).strip()
        flag   = str(row["Flag"]).strip()

        message_id = f"bulk-row-{idx}"
        json_path  = workspace / f"{message_id}.json"

        if json_path.exists():
            logger.info("[%d/%d] Skipping (state file exists): %s", idx + 1, total_rows, se_pn)
            continue

        customer_request = _build_customer_request(se_pn, se_man, flag)
        init_state(message_id, customer_request, workspace)
        logger.info("[%d/%d] Queued: %s from %s (%s)", idx + 1, total_rows, se_pn, se_man, flag)

        queued += 1
        if LIMIT_RUN and queued >= LIMIT_RUN_COUNT:
            logger.info("[LIMIT_RUN] Queued %d rows. Stopping queue phase.", LIMIT_RUN_COUNT)
            break

    logger.info("Running pipeline across %d message(s) in workspace...", queued or "all")
    completed, failed = run_pipeline(workspace)

    logger.info("Pipeline done. Writing CSV output...")
    _write_csv_results(df, completed, failed, OUTPUT_CSV)
    logger.info(
        "Done. %d completed, %d failed. Results at: %s",
        len(completed), len(failed), OUTPUT_CSV,
    )


if __name__ == "__main__":
    run_bulk()
