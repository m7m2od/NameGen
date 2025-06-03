"""
WaveNet Character-Level Language Model Package

This package contains a PyTorch implementation of a WaveNet-inspired
character-level language model for name generation.
"""

from .layers import Linear, BatchNorm, Tanh, Embedding, FlattenConsecutive, Sequential
from .model import WaveNetCharModel, Trainer, Evaluator, NameGenerator, create_model_from_data
from .data_utils import prepare_data, load_names, create_character_mappings, build_dataset

__version__ = "1.0.0"
__author__ = "WaveNet Character-LM Team"

__all__ = [
    # Layers
    "Linear",
    "BatchNorm", 
    "Tanh",
    "Embedding",
    "FlattenConsecutive",
    "Sequential",
    
    # Model components
    "WaveNetCharModel",
    "Trainer",
    "Evaluator", 
    "NameGenerator",
    "create_model_from_data",
    
    # Data utilities
    "prepare_data",
    "load_names",
    "create_character_mappings",
    "build_dataset",
]