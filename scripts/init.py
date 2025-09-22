"""
Project Aether: Core scripts for distributed reasoning model training
"""

__version__ = "1.0.0"
__author__ = "Project Aether Team"

from .model import ReasoningTransformer
from .merging import merge_contributions
from .checkpointing import save_checkpoint, load_checkpoint
from .dataset_loading import load_shard, preprocess_dataset
from .data_generation import generate_reasoning_traces
from .training_utils import train_shard, evaluate_model

__all__ = [
    "ReasoningTransformer",
    "merge_contributions",
    "save_checkpoint",
    "load_checkpoint",
    "load_shard",
    "preprocess_dataset",
    "generate_reasoning_traces",
    "train_shard",
    "evaluate_model"
]