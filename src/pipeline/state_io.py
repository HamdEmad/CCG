"""
Pipeline workspace state I/O.

Each customer message is represented as one JSON file in the workspace
directory. The JSON holds all keys for all pipeline stages. Each stage
reads only the keys it needs and writes only the keys it produces.

If a stage's status is already "done" or "skipped" when the orchestrator
scans the workspace, that stage is skipped for that message -- this is
what enables resuming interrupted runs and injecting pre-extracted data.

File layout:
  workspace/
    {message_id}.json              ← full state for one message
    {message_id}_part_0_scrape.txt ← scraped text, stored separately
    {message_id}_part_1_scrape.txt

Status values used in every stage key:
  "pending"  — not yet processed, orchestrator should run this stage
  "done"     — completed successfully
  "failed"   — failed, will not be retried
  "skipped"  — intentionally bypassed (e.g., scrape skipped because
                customer URL already gave a result)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.state import PartDetails

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blank_part_stages() -> dict:
    """Return a fresh stage-status block for one part — all pending."""
    return {
        "customer_url":  {"status": "pending"},
        "search":        {"status": "pending"},
        "filter":        {"status": "pending"},
        "scrape":        {"status": "pending"},
        "url_inference": {"status": "pending"},
        "scrape_inferred": {"status": "pending"},
        "browser":       {"status": "pending"},
        "attrs":         {"status": "pending"},
    }


def _part_record(part_id: str, part_details: dict) -> dict:
    """Build a fresh part record to embed in a message state file."""
    record = {"part_id": part_id, "part_details": part_details}
    record.update(_blank_part_stages())
    record["final_status"] = "pending"
    record["final_result"] = None
    return record


# ---------------------------------------------------------------------------
# State file creation
# ---------------------------------------------------------------------------

def init_state(message_id: str, customer_message: str, workspace: Path) -> Path:
    """
    Create a fresh JSON state file for a new customer message.

    The extraction stage is set to pending; no parts are populated yet.
    Returns the path to the created file.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    state = {
        "message_id": message_id,
        "customer_message": customer_message,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "extraction": {"status": "pending"},
        "parts": [],
    }
    path = workspace / f"{message_id}.json"
    save_state(path, state)
    logger.debug("Initialized state for message %s at %s", message_id, path)
    return path


def inject_pre_extracted(
    message_id: str,
    parts: list[PartDetails],
    workspace: Path,
    customer_message: str = "",
) -> Path:
    """
    Create a JSON state file for a message where extraction is already done.

    Sets extraction.status = 'done' and populates the parts list with all
    stages set to 'pending'. The orchestrator will skip the extraction
    stage for this message and start from web search.

    Use this when part lists come from an external source (e.g., a
    database, a pre-parsed Excel sheet, or a previous run's output).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    part_records = [
        _part_record(f"part_{i}", part.model_dump())
        for i, part in enumerate(parts)
    ]
    state = {
        "message_id": message_id,
        "customer_message": customer_message,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "extraction": {"status": "done"},
        "parts": part_records,
    }
    path = workspace / f"{message_id}.json"
    save_state(path, state)
    logger.debug(
        "Injected pre-extracted state for message %s (%d parts)", message_id, len(parts)
    )
    return path


# ---------------------------------------------------------------------------
# Atomic save / load
# ---------------------------------------------------------------------------

def save_state(json_path: Path, state: dict) -> None:
    """
    Atomically write state to disk.

    Writes to a .tmp file first, then os.replace() renames it over the
    real file. This guarantees that a crash mid-write leaves either the
    old file or the new file intact — never a half-written file.
    """
    state["updated_at"] = _now_iso()
    tmp_path = json_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(state, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_path, json_path)


def load_state(json_path: Path) -> dict:
    """Load and parse a state JSON file."""
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Scraped text sidecar files
# ---------------------------------------------------------------------------

def get_scrape_text_path(json_path: Path, part_id: str) -> Path:
    """Return the sidecar .txt path for a part's scraped page text."""
    return json_path.parent / f"{json_path.stem}_{part_id}_scrape.txt"


def write_scrape_text(json_path: Path, part_id: str, text: str) -> None:
    """Write scraped text to its sidecar file (NOT inside the JSON)."""
    path = get_scrape_text_path(json_path, part_id)
    path.write_text(text, encoding="utf-8")


def read_scrape_text(json_path: Path, part_id: str) -> str | None:
    """
    Read scraped text from the sidecar file.
    Returns None if the file doesn't exist.
    """
    path = get_scrape_text_path(json_path, part_id)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Workspace scanning helpers
# ---------------------------------------------------------------------------

def reset_failed_parts_to_pending(workspace: Path, message_ids: list[str] | None = None) -> None:
    """
    Scan state files and reset any failed parts back to 'pending'.
    This clears 'failed' and 'skipped' stage statuses so the orchestrator can retry them.
    """
    count = 0
    for json_path in sorted(workspace.glob("*.json")):
        if message_ids is not None and json_path.stem not in message_ids:
            continue
        try:
            state = load_state(json_path)
        except Exception:
            continue
            
        changed = False
        for part in state.get("parts", []):
            if part.get("final_status") == "failed":
                part["final_status"] = "pending"
                part["final_result"] = None
                changed = True
                count += 1
                for stage_key in ["customer_url", "search", "filter", "scrape", "url_inference", "scrape_inferred", "browser", "attrs"]:
                    if stage_key in part:
                        st = part[stage_key].get("status")
                        if st in ("failed", "skipped"):
                            part[stage_key]["status"] = "pending"
                            
        if changed:
            save_state(json_path, state)
            
    if count > 0:
        logger.info("Reset %d failed parts back to pending", count)


def collect_parts_at_stage(
    workspace: Path, stage: str, message_ids: list[str] | None = None
) -> list[tuple[Path, dict, dict]]:
    """
    Scan all JSON state files in workspace and return every part whose
    given stage has status == 'pending'.

    Returns a list of (json_path, state, part_record) tuples.
    The caller can mutate part_record in place, then call save_state() to
    persist changes — state is a live reference to the same dict.
    """
    results = []
    for json_path in sorted(workspace.glob("*.json")):
        if message_ids is not None and json_path.stem not in message_ids:
            continue
        try:
            state = load_state(json_path)
        except Exception as e:
            logger.warning("Skipping unreadable state file %s: %s", json_path.name, e)
            continue

        for part in state.get("parts", []):
            stage_data = part.get(stage, {})
            if stage_data.get("status") == "pending":
                results.append((json_path, state, part))

    logger.debug(
        "Stage '%s': found %d pending part(s) across workspace", stage, len(results)
    )
    return results


def collect_messages_at_stage(
    workspace: Path, stage: str, message_ids: list[str] | None = None
) -> list[tuple[Path, dict]]:
    """
    Scan all JSON state files and return messages where the top-level
    stage key has status == 'pending'.

    Used for the extraction stage, which operates on the message as a
    whole (not per-part) and populates the parts list.
    """
    results = []
    for json_path in sorted(workspace.glob("*.json")):
        if message_ids is not None and json_path.stem not in message_ids:
            continue
        try:
            state = load_state(json_path)
        except Exception as e:
            logger.warning("Skipping unreadable state file %s: %s", json_path.name, e)
            continue

        stage_data = state.get(stage, {})
        if stage_data.get("status") == "pending":
            results.append((json_path, state))

    logger.debug(
        "Stage '%s': found %d pending message(s) across workspace", stage, len(results)
    )
    return results


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_final_results(workspace: Path) -> tuple[list[dict], list[dict]]:
    """
    Walk all JSON state files and aggregate final results.

    Returns:
        completed: list of final_result dicts for parts with final_status == "completed"
        failed:    list of final_result dicts for parts with final_status == "failed"
    """
    completed: list[dict] = []
    failed: list[dict] = []

    for json_path in sorted(workspace.glob("*.json")):
        try:
            state = load_state(json_path)
        except Exception as e:
            logger.warning("Skipping unreadable state file %s: %s", json_path.name, e)
            continue

        for part in state.get("parts", []):
            final_status = part.get("final_status", "pending")
            final_result = part.get("final_result") or {}
            # Merge part_details into final_result for downstream consumers
            result = {
                "part_details": part.get("part_details", {}),
                **final_result,
            }
            if final_status == "completed":
                completed.append(result)
            elif final_status == "failed":
                failed.append(result)
            # "pending" means a stage is still running or stuck — leave for next run

    return completed, failed
