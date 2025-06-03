"""
Data processing utilities for the WaveNet character-level language model.

This module handles loading, preprocessing, and splitting of text data
for training character-level language models.
"""

import torch
import random
from typing import List, Tuple, Dict


def load_names(filepath: str) -> List[str]:
    """Load names from a text file.
    
    Args:
        filepath: Path to the text file containing names (one per line)
        
    Returns:
        List of names as strings
    """
    with open(filepath, 'r') as f:
        return f.read().splitlines()


def create_character_mappings(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Create character-to-index and index-to-character mappings.
    
    Args:
        words: List of words to extract characters from
        
    Returns:
        Tuple of (string_to_index, index_to_string, vocab_size)
    """
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0  # Special end-of-word token
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)
    
    return stoi, itos, vocab_size


def build_dataset(words: List[str], stoi: Dict[str, int], block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build dataset of context-target pairs for training.
    
    Args:
        words: List of words to process
        stoi: String-to-index mapping
        block_size: Size of context window
        
    Returns:
        Tuple of (contexts, targets) as tensors
    """
    X, Y = [], []
    
    for word in words:
        context = [0] * block_size  # Start with padding tokens
        for ch in word + '.':  # Add end-of-word token
            ix = stoi[ch]
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]  # Sliding window
    
    return torch.tensor(X), torch.tensor(Y)


def split_dataset(words: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """Split dataset into train, validation, and test sets.
    
    Args:
        words: List of words to split
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        
    Returns:
        Tuple of (train_words, val_words, test_words)
    """
    random.shuffle(words)
    
    n_total = len(words)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    train_words = words[:n_train]
    val_words = words[n_train:n_train + n_val]
    test_words = words[n_train + n_val:]
    
    return train_words, val_words, test_words


def prepare_data(filepath: str, block_size: int = 3, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Complete data preparation pipeline.
    
    Args:
        filepath: Path to the names file
        block_size: Size of context window
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        
    Returns:
        Dictionary containing all prepared data and mappings
    """
    # Load and prepare data
    words = load_names(filepath)
    stoi, itos, vocab_size = create_character_mappings(words)
    
    # Split data
    train_words, val_words, test_words = split_dataset(words, train_ratio, val_ratio)
    
    # Build datasets
    X_train, Y_train = build_dataset(train_words, stoi, block_size)
    X_val, Y_val = build_dataset(val_words, stoi, block_size)
    X_test, Y_test = build_dataset(test_words, stoi, block_size)
    
    return {
        'datasets': {
            'train': (X_train, Y_train),
            'val': (X_val, Y_val),
            'test': (X_test, Y_test)
        },
        'mappings': {
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size
        },
        'config': {
            'block_size': block_size,
            'train_size': len(train_words),
            'val_size': len(val_words),
            'test_size': len(test_words)
        }
    }


def print_sample_data(X: torch.Tensor, Y: torch.Tensor, itos: Dict[int, str], n_samples: int = 5):
    """Print sample context-target pairs for inspection.
    
    Args:
        X: Context tensor
        Y: Target tensor
        itos: Index-to-string mapping
        n_samples: Number of samples to print
    """
    print("Sample context-target pairs:")
    print("-" * 40)
    
    for i in range(min(n_samples, len(X))):
        try:
            context_chars = [itos[ix.item()] for ix in X[i]]
            target_char = itos[Y[i].item()]
            context_str = ''.join(context_chars).replace('.', '·')  # Make dots visible
            print(f"Context: '{context_str}' -> Target: '{target_char}'")
        except KeyError as e:
            print(f"KeyError for sample {i}: {e}")