import os

from datasets import Dataset

def to_arrow(load_path: str = "training_set.jsonl", save_path: str = "agent_dataset") -> None:
    """Given a path to a training dataset, convert it to Avro format.

    Args:
        load_path (str): The path to the training dataset.
        save_path (str): The path to save the converted dataset.
    """
    Dataset.from_json(load_path).save_to_disk(save_path)


def load_dataset(path: str = "agent_dataset", source: str = "training_set.jsonl") -> Dataset:
    """Load the Arrow cache at `path`, building it from `source` (jsonl) if missing."""
    if not os.path.exists(path):
        to_arrow(load_path=source, save_path=path)

    return Dataset.load_from_disk(path)
