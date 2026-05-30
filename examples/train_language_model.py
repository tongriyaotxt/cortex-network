"""
Example: Training CORTEX on a toy language modeling task.

This demonstrates how to use CORTEX for sequence prediction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cortex.cortex_model import CORTEXModel


class ToySequenceDataset(Dataset):
    """Toy dataset: predict next token in repeating pattern."""
    def __init__(self, seq_len=32, vocab_size=100, num_samples=1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        
        # Create deterministic patterns
        torch.manual_seed(42)
        self.patterns = [
            torch.randint(0, vocab_size, (seq_len + 1,))
            for _ in range(10)
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx % len(self.patterns)]
        return pattern[:-1], pattern[1:]  # input, target


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, labels=targets)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, labels=targets)
            loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    print("CORTEX Language Modeling Example")
    print("=" * 60)
    
    # Config
    vocab_size = 100
    d_model = 126
    n_layers = 4
    seq_len = 32
    batch_size = 16
    epochs = 5
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print(f"\nCreating CORTEX model:")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  vocab_size: {vocab_size}")
    
    model = CORTEXModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_modules=4,
        workspace_dim=63,
        max_seq_len=seq_len,
        dropout=0.1,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create datasets
    train_dataset = ToySequenceDataset(seq_len=seq_len, vocab_size=vocab_size, num_samples=1000)
    val_dataset = ToySequenceDataset(seq_len=seq_len, vocab_size=vocab_size, num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Sample generation
        model.eval()
        with torch.no_grad():
            sample_input = torch.randint(0, vocab_size, (1, 8)).to(device)
            generated = model.generate(
                sample_input,
                max_new_tokens=10,
                temperature=0.8,
            )
            print(f"Sample input: {sample_input[0].tolist()}")
            print(f"Generated: {generated[0].tolist()}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
