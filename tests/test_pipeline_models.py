import pytest
from pathlib import Path
import tempfile
from pipeline.config import Settings
from pipeline.state import PartDetails, PartStatus, PartAttributeResult, PipelineResult
from pipeline.state_io import init_state, load_state, save_state

def test_settings_initialization():
    settings = Settings(
        llm_model_name="test-model",
        search_provider="duckduckgo",
        browser_headless=True,
    )
    assert settings.llm_model_name == "test-model"
    assert settings.search_provider == "duckduckgo"
    assert settings.browser_headless is True

def test_part_details_model():
    part = PartDetails(
        part="TEST-1234",
        manufacturer="Acme Corp",
        message="Requesting info for TEST-1234",
    )
    assert part.part == "TEST-1234"
    assert part.manufacturer == "Acme Corp"
    assert part.part_series is None

def test_pipeline_state_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        message_id = "test_msg_001"
        message_text = "Sample customer message for part XYZ"

        # Initialize state file
        state_file = init_state(message_id, message_text, workspace)
        assert state_file.exists()

        # Load state file
        state = load_state(state_file)
        assert state["message_id"] == message_id
        assert state["customer_message"] == message_text
        assert state["parts"] == []
        assert state["extraction"]["status"] == "pending"
