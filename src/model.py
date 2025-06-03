"""
WaveNet Character-Level Language Model

This module contains the model architecture, training utilities,
and generation functions for the character-level language model.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from .layers import Sequential, Embedding, FlattenConsecutive, Linear, BatchNorm, Tanh


class WaveNetCharModel:
    """WaveNet-style character-level language model."""
    
    def __init__(self, vocab_size: int, block_size: int, n_embd: int = 10, n_hidden: int = 200):
        """Initialize the model.
        
        Args:
            vocab_size: Size of the character vocabulary
            block_size: Size of the context window
            n_embd: Embedding dimension
            n_hidden: Hidden layer dimension
        """
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        
        # Build the model architecture
        self.model = Sequential([
            Embedding(vocab_size, n_embd),
            FlattenConsecutive(block_size),
            Linear(n_embd * block_size, n_hidden),
            BatchNorm(n_hidden),
            Tanh(),
            Linear(n_hidden, vocab_size)
        ])
        
        # Initialize the final layer with smaller weights
        with torch.no_grad():
            self.model.layers[-1].weight *= 0.1
        
        # Get all parameters and enable gradients
        self.params = self.model.parameters()
        for param in self.params:
            param.requires_grad = True
        
        print(f"Model initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count the total number of parameters in the model."""
        return sum(param.nelement() for param in self.params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def set_training(self, training: bool):
        """Set training mode for all layers."""
        for layer in self.model.layers:
            if hasattr(layer, 'training'):
                layer.training = training


class Trainer:
    """Training utilities for the WaveNet model."""
    
    def __init__(self, model: WaveNetCharModel, learning_rates: Dict[str, float] = None):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            learning_rates: Dictionary with 'initial' and 'final' learning rates
        """
        self.model = model
        self.learning_rates = learning_rates or {'initial': 0.1, 'final': 0.01}
        self.loss_history = []
    
    def get_learning_rate(self, step: int, warmup_steps: int = 1000) -> float:
        """Get learning rate for the current step."""
        if step < warmup_steps:
            return self.learning_rates['initial']
        else:
            return self.learning_rates['final']
    
    def train_step(self, X: torch.Tensor, Y: torch.Tensor, step: int) -> float:
        """Perform one training step."""
        # Forward pass
        logits = self.model.forward(X)
        loss = F.cross_entropy(logits, Y)
        
        # Backward pass
        for param in self.model.params:
            param.grad = None
        loss.backward()
        
        # Update parameters
        lr = self.get_learning_rate(step)
        for param in self.model.params:
            param.data += -lr * param.grad
        
        return loss.item()
    
    def train(self, X_train: torch.Tensor, Y_train: torch.Tensor, 
              max_steps: int = 100000, batch_size: int = 32, 
              log_interval: int = 10000) -> List[float]:
        """Train the model.
        
        Args:
            X_train: Training input data
            Y_train: Training target data
            max_steps: Maximum number of training steps
            batch_size: Batch size for training
            log_interval: Steps between logging
            
        Returns:
            List of loss values
        """
        print(f"Starting training for {max_steps:,} steps...")
        
        for step in range(max_steps):
            # Sample a batch
            ix = torch.randint(0, X_train.shape[0], (batch_size,))
            X_batch = X_train[ix]
            Y_batch = Y_train[ix]
            
            # Training step
            loss = self.train_step(X_batch, Y_batch, step)
            self.loss_history.append(loss)
            
            # Logging
            if step % log_interval == 0:
                print(f"Step {step:,}: loss {loss:.4f}")
        
        print("Training completed!")
        return self.loss_history
    
    def plot_loss(self, smooth_factor: int = 1000):
        """Plot the training loss curve."""
        if not self.loss_history:
            print("No training history available.")
            return
        
        # Smooth the loss curve
        loss_tensor = torch.tensor(self.loss_history)
        if len(loss_tensor) >= smooth_factor:
            smoothed_loss = loss_tensor.view(-1, smooth_factor).mean(1)
            plt.figure(figsize=(10, 6))
            plt.plot(smoothed_loss)
            plt.title('Training Loss')
            plt.xlabel(f'Steps (x{smooth_factor})')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_history)
            plt.title('Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()


class Evaluator:
    """Evaluation utilities for the model."""
    
    def __init__(self, model: WaveNetCharModel):
        self.model = model
    
    @torch.no_grad()
    def evaluate_split(self, X: torch.Tensor, Y: torch.Tensor, split_name: str = "eval") -> float:
        """Evaluate the model on a data split.
        
        Args:
            X: Input data
            Y: Target data
            split_name: Name of the split (for logging)
            
        Returns:
            Loss value
        """
        # Set to evaluation mode
        self.model.set_training(False)
        
        logits = self.model.forward(X)
        loss = F.cross_entropy(logits, Y)
        loss_val = loss.item()
        
        print(f"{split_name} loss: {loss_val:.4f}")
        return loss_val
    
    def evaluate_all_splits(self, datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate on all data splits.
        
        Args:
            datasets: Dictionary containing 'train', 'val', 'test' splits
            
        Returns:
            Dictionary of loss values for each split
        """
        results = {}
        for split_name, (X, Y) in datasets.items():
            results[split_name] = self.evaluate_split(X, Y, split_name)
        return results


class NameGenerator:
    """Name generation utilities."""
    
    def __init__(self, model: WaveNetCharModel, itos: Dict[int, str]):
        self.model = model
        self.itos = itos
    
    @torch.no_grad()
    def generate_name(self, max_length: int = 50) -> str:
        """Generate a single name.
        
        Args:
            max_length: Maximum length of generated name
            
        Returns:
            Generated name as a string
        """
        # Set to evaluation mode
        self.model.set_training(False)
        
        out = []
        context = [0] * self.model.block_size  # Start with padding tokens
        
        for _ in range(max_length):
            # Get predictions
            logits = self.model.forward(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            
            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            
            # Update context
            context = context[1:] + [ix]
            out.append(ix)
            
            # Stop if we hit the end token
            if ix == 0:
                break
        
        # Convert indices to characters
        name = ''.join(self.itos[i] for i in out[:-1])  # Exclude the end token
        return name
    
    def generate_names(self, n_names: int = 20, max_length: int = 50) -> List[str]:
        """Generate multiple names.
        
        Args:
            n_names: Number of names to generate
            max_length: Maximum length of each name
            
        Returns:
            List of generated names
        """
        names = []
        for _ in range(n_names):
            name = self.generate_name(max_length)
            names.append(name)
        return names
    
    def print_generated_names(self, n_names: int = 20):
        """Generate and print names."""
        names = self.generate_names(n_names)
        print(f"\nGenerated {n_names} names:")
        print("-" * 30)
        for i, name in enumerate(names, 1):
            print(f"{i:2d}. {name}")


def create_model_from_data(data_dict: Dict, n_embd: int = 10, n_hidden: int = 200) -> WaveNetCharModel:
    """Create a model from prepared data dictionary.
    
    Args:
        data_dict: Dictionary from prepare_data function
        n_embd: Embedding dimension
        n_hidden: Hidden layer dimension
        
    Returns:
        Initialized WaveNetCharModel
    """
    vocab_size = data_dict['mappings']['vocab_size']
    block_size = data_dict['config']['block_size']
    
    return WaveNetCharModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_hidden=n_hidden
    )