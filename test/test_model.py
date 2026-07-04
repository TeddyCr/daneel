import pytest
import torch

from ..model import load_model, load_tokenizer, lora_config


@pytest.fixture(scope="session")
def device() -> str:
    """Determine the appropriate device for the model.

    Returns:
        str: The device to be used for the model.
    """    
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def test_load_model(device: str):
    """Test the load_model function to ensure it loads the model correctly."""
    model = load_model(model_id="HuggingFaceTB/SmolLM2-135M", device=device)
    assert model is not None, "Failed to load the model."

def test_load_tokenizer(device: str):
    """Test the load_tokenizer function to ensure it loads the tokenizer correctly."""
    tokenizer = load_tokenizer(model_id="HuggingFaceTB/SmolLM2-135M", device_map=device)
    assert tokenizer is not None, "Failed to load the tokenizer."

def test_lora_config():
    """Test the lora_config function to ensure it returns a valid configuration."""
    config = lora_config()
    assert config is not None, "Failed to create LoRA configuration."