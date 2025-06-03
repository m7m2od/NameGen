#!/usr/bin/env python3
"""
Main training script for WaveNet Character-Level Language Model

This script handles the complete training pipeline including data preparation,
model training, evaluation, and name generation.
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_utils import prepare_data, print_sample_data
from model import WaveNetCharModel, Trainer, Evaluator, NameGenerator, create_model_from_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train WaveNet Character-Level Language Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/names.txt',
                       help='Path to the names file')
    parser.add_argument('--block_size', type=int, default=3,
                       help='Context window size')
    
    # Model arguments
    parser.add_argument('--n_embd', type=int, default=10,
                       help='Embedding dimension')
    parser.add_argument('--n_hidden', type=int, default=200,
                       help='Hidden layer dimension')
    
    # Training arguments
    parser.add_argument('--max_steps', type=int, default=100000,
                       help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr_initial', type=float, default=0.1,
                       help='Initial learning rate')
    parser.add_argument('--lr_final', type=float, default=0.01,
                       help='Final learning rate')
    parser.add_argument('--log_interval', type=int, default=10000,
                       help='Steps between logging')
    
    # Output arguments
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--generate_names', type=int, default=20,
                       help='Number of names to generate after training')
    parser.add_argument('--plot_loss', action='store_true',
                       help='Plot training loss curve')
    
    return parser.parse_args()


def save_model_and_config(model, data_dict, args, model_dir):
    """Save model parameters and configuration."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model parameters
    model_path = os.path.join(model_dir, 'model_params.pt')
    torch.save({
        'params': [p.data for p in model.params],
        'model_config': {
            'vocab_size': model.vocab_size,
            'block_size': model.block_size,
            'n_embd': model.n_embd,
            'n_hidden': model.n_hidden
        }
    }, model_path)
    
    # Save mappings and config
    config_path = os.path.join(model_dir, 'config.json')
    config = {
        'mappings': data_dict['mappings'],
        'training_config': vars(args),
        'data_stats': data_dict['config']
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Config saved to {config_path}")


def main():
    """Main training function."""
    args = parse_arguments()
    print("WaveNet Character-Level Language Model Training")
    print("=" * 50)
    
    # Prepare data
    print(f"Loading data from {args.data_path}...")
    data_dict = prepare_data(
        filepath=args.data_path,
        block_size=args.block_size,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    # Print data statistics
    config = data_dict['config']
    mappings = data_dict['mappings']
    print(f"Data loaded successfully!")
    print(f"Vocabulary size: {mappings['vocab_size']}")
    print(f"Block size: {config['block_size']}")
    print(f"Training samples: {config['train_size']:,}")
    print(f"Validation samples: {config['val_size']:,}")
    print(f"Test samples: {config['test_size']:,}")
    
    # Show sample data
    X_train, Y_train = data_dict['datasets']['train']
    print_sample_data(X_train, Y_train, mappings['itos'], n_samples=5)
    
    # Create model
    print(f"\nCreating model...")
    model = create_model_from_data(data_dict, args.n_embd, args.n_hidden)
    print(f"Model architecture:")
    print(model.model)
    
    # Create trainer
    learning_rates = {
        'initial': args.lr_initial,
        'final': args.lr_final
    }
    trainer = Trainer(model, learning_rates)
    
    # Train the model
    print(f"\nStarting training...")
    loss_history = trainer.train(
        X_train, Y_train,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval
    )
    
    # Plot loss if requested
    if args.plot_loss:
        trainer.plot_loss()
    
    # Evaluate on all splits
    print(f"\nEvaluating model...")
    evaluator = Evaluator(model)
    results = evaluator.evaluate_all_splits(data_dict['datasets'])
    
    # Save model and configuration
    save_model_and_config(model, data_dict, args, args.model_dir)
    
    # Generate sample names
    if args.generate_names > 0:
        print(f"\nGenerating sample names...")
        generator = NameGenerator(model, mappings['itos'])
        generator.print_generated_names(args.generate_names)
    
    print(f"\nTraining completed successfully!")
    print(f"Final results:")
    for split, loss in results.items():
        print(f"  {split.capitalize()} loss: {loss:.4f}")


if __name__ == "__main__":
    main()