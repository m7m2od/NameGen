#!/usr/bin/env python3
"""
Basic Usage Example for WaveNet Character-Level Language Model

This example demonstrates how to use the WaveNet model for name generation
with minimal setup. Perfect for getting started quickly!
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_utils import prepare_data
from model import create_model_from_data, Trainer, NameGenerator

def main():
    print("🚀 WaveNet Name Generator - Basic Example")
    print("=" * 45)
    
    # Step 1: Prepare the data
    print("📊 Loading and preparing data...")
    data_path = Path(__file__).parent.parent / 'data' / 'names.txt'
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please make sure names.txt is in the data/ directory")
        return
    
    data_dict = prepare_data(str(data_path), block_size=3)
    print(f"✅ Loaded {data_dict['config']['train_size']} training examples")
    
    # Step 2: Create and train the model (quick version)
    print("\n🧠 Creating and training model...")
    model = create_model_from_data(data_dict, n_embd=8, n_hidden=100)
    
    # Quick training with fewer steps for demo
    trainer = Trainer(model, {'initial': 0.1, 'final': 0.01})
    X_train, Y_train = data_dict['datasets']['train']
    
    print("🏃‍♂️ Quick training (10K steps)...")
    trainer.train(X_train, Y_train, max_steps=10000, batch_size=32, log_interval=2500)
    
    # Step 3: Generate some names!
    print("\n🎯 Generating new names...")
    generator = NameGenerator(model, data_dict['mappings']['itos'])
    
    names = generator.generate_names(15)
    print("\n✨ Generated Names:")
    print("-" * 20)
    for i, name in enumerate(names, 1):
        print(f"{i:2d}. {name.capitalize()}")
    
    print(f"\n🎉 Done! Generated {len(names)} unique names")
    print("\nTip: Run the full training script for better results:")
    print("  python src/train.py --max_steps 100000")

if __name__ == "__main__":
    main()