import json
import os
import shutil
import pytest
from datasets import Dataset
from ..dataset import to_arrow

JSONL_PATH = "./test_dataset.jsonl"
DATASET_PATH = "./agent_dataset"

SAMPLE_RECORDS = [
    {"instruction": "What is 2+2?", "response": "4"},
    {"instruction": "What is the capital of France?", "response": "Paris"},
]

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    with open(JSONL_PATH, "w") as f:
        for record in SAMPLE_RECORDS:
            f.write(json.dumps(record) + "\n")

    yield

    if os.path.exists(JSONL_PATH):
        os.remove(JSONL_PATH)
    if os.path.exists(DATASET_PATH):
        shutil.rmtree(DATASET_PATH)

def test_to_arrow():
    """Test the to_arrow function to ensure it converts data to Avro format correctly."""
    to_arrow(JSONL_PATH)

    assert os.path.exists(DATASET_PATH), "Dataset directory was not created."

    dataset = Dataset.load_from_disk(DATASET_PATH)
    assert len(dataset) == len(SAMPLE_RECORDS), "Record count mismatch."
    assert dataset[0]["instruction"] == SAMPLE_RECORDS[0]["instruction"]
    assert dataset[0]["response"] == SAMPLE_RECORDS[0]["response"]
