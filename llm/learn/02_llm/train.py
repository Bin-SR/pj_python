# -*- coding: utf-8 -*-
'''
02_llm/train.py - MiniGPT Training Script

Trains MiniGPT on the TinyShakespeare dataset (or any text file).

Key training features:
- Gradient accumulation (simulate larger batch sizes)
- Mixed precision training (FP16) for memory efficiency
- Learning rate warmup + cosine decay
- Gradient clipping
- Periodic checkpointing + text generation samples

RTX 3050 tips:
- Use config_tiny() or config_small() for comfortable VRAM usage
- If OOM: reduce batch_size, use gradient_accumulation_steps
- Enable mixed precision (AMP) for ~30% memory savings

Usage:
    python train.py --config small --epochs 10 --batch_size 8
'''

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import time
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_foundations'))

from mini_gpt import MiniGPT, MiniGPTConfig, config_tiny, config_small, config_medium
from tokenizer import BPETokenizer


# ============================================================
# Dataset utilities
# ============================================================
def load_tinyshakespeare(data_dir: str = './data') -> str:
    '''
    Download and load the TinyShakespeare dataset.
    A small (~1MB) text file, perfect for learning.
    '''
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'tinyshakespeare.txt')

    if not os.path.exists(path):
        print('Downloading TinyShakespeare dataset...')
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, path)
        print(f'Downloaded to {path}')

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f'Dataset loaded: {len(text)} characters')
    return text


def create_dataloader(
    text: str,
    tokenizer: BPETokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
):
    '''
    Convert text to token IDs and create a simple batch iterator.

    For simplicity, we use sequential chunking (no shuffling).
    For production, use PyTorch DataLoader with proper shuffling.
    '''
    # Encode full text
    ids = tokenizer.encode(text, add_special_tokens=False)
    data = torch.tensor(ids, dtype=torch.long)

    # Split into (input, target) pairs shifted by 1
    num_batches = (len(data) - 1) // (batch_size * seq_len)

    for i in range(num_batches):
        start = i * batch_size * seq_len
        x = data[start:start + batch_size * seq_len].view(batch_size, seq_len)
        y = data[start + 1:start + 1 + batch_size * seq_len].view(batch_size, seq_len)
        yield x.to(device), y.to(device)


# ============================================================
# Training loop
# ============================================================
class Trainer:
    '''
    GPT Trainer with best practices:
    - Gradient accumulation
    - Mixed precision (AMP)
    - LR warmup + cosine schedule
    - Periodic evaluation (perplexity + sample generation)
    '''

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps

        # Optimizer
        self.optimizer = model.configure_optimizers(learning_rate, weight_decay)
        self.base_lr = learning_rate

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=(device.type == 'cuda'))

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def _lr_schedule(self, step: int, total_steps: int) -> float:
        '''Cosine learning rate schedule with linear warmup.'''
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        elif step > total_steps:
            return self.base_lr * 0.1  # Minimum LR
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_step(self, x: torch.Tensor, y: torch.Tensor, step: int, total_steps: int) -> float:
        '''Single training step with gradient accumulation.'''
        self.model.train()

        # Update LR
        lr = self._lr_schedule(step, total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass with mixed precision
        with autocast(enabled=(self.device.type == 'cuda')):
            logits = self.model(x)  # (batch, seq_len, vocab_size)
            # Cross-entropy loss: flatten to (batch*seq_len, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Update weights after accumulation steps
        if (step + 1) % self.grad_accum_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 10) -> float:
        '''Compute validation loss.'''
        self.model.eval()
        total_loss = 0.0
        count = 0

        for x, y in val_loader:
            if count >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with autocast(enabled=(self.device.type == 'cuda')):
                logits = self.model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                )
            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    @torch.no_grad()
    def generate_sample(self, prompt: str = 'The ', max_tokens: int = 100) -> str:
        '''Generate a text sample for qualitative evaluation.'''
        self.model.eval()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        generated = self.model.generate(
            input_ids, max_new_tokens=max_tokens, temperature=0.8, top_k=40
        )

        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, path: str, epoch: int, step: int):
        '''Save model checkpoint.'''
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f'Checkpoint saved to {path}')


# ============================================================
# Main training function
# ============================================================
def train(
    config_name: str = 'small',
    batch_size: int = 8,
    seq_len: int = 128,
    epochs: int = 5,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    save_dir: str = './checkpoints',
    data_dir: str = './data',
):
    '''
    Main training function.

    Args:
        config_name: 'tiny', 'small', or 'medium'
        batch_size: Per-GPU batch size (effective batch = batch_size * grad_accum_steps)
        seq_len: Sequence length for training
        epochs: Number of training epochs
        grad_accum_steps: Gradient accumulation steps
        learning_rate: Peak learning rate
        save_dir: Directory for checkpoints
        data_dir: Directory for datasets
    '''
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

    # Config
    configs = {'tiny': config_tiny, 'small': config_small, 'medium': config_medium}
    config = configs[config_name]()
    print(f'Model config: {config}')

    # Tokenizer
    tokenizer = BPETokenizer()

    # Load data
    text = load_tinyshakespeare(data_dir)

    # Train tokenizer on the corpus
    tokenizer.train(text, vocab_size=config.vocab_size, min_freq=2)

    # Update config vocab_size to match tokenizer
    config.vocab_size = tokenizer.vocab_size

    # Create model
    model = MiniGPT(config)

    # Split data (90% train, 10% val)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create trainer
    trainer = Trainer(
        model, tokenizer, device,
        learning_rate=learning_rate,
        grad_accum_steps=grad_accum_steps,
        warmup_steps=50,
    )

    # Training loop
    total_steps = epochs * (len(train_text) // (batch_size * seq_len))
    print(f'Training for {epochs} epochs, ~{total_steps} steps')
    print(f'Effective batch size: {batch_size * grad_accum_steps}')

    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loader = create_dataloader(train_text, tokenizer, batch_size, seq_len, device)

        for x, y in train_loader:
            loss = trainer.train_step(x, y, global_step, total_steps)
            trainer.train_losses.append(loss)
            global_step += 1

            if global_step % 50 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                print(
                    f'  Step {global_step:5d}/{total_steps} | '
                    f'Loss: {loss:.4f} | '
                    f'LR: {trainer.optimizer.param_groups[0]["lr"]:.2e} | '
                    f'{steps_per_sec:.1f} steps/s'
                )

            if global_step % 200 == 0:
                # Generate sample
                sample = trainer.generate_sample(prompt='First Citizen: ')
                print(f'  --- Sample ---')
                print(f'  {sample[:200]}...')
                print(f'  -------------')

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch + 1}/{epochs} completed in {epoch_time:.1f}s')

        # Save checkpoint
        trainer.save_checkpoint(
            os.path.join(save_dir, f'minigpt_{config_name}_epoch{epoch + 1}.pt'),
            epoch, global_step
        )

    total_time = time.time() - start_time
    print(f'Training complete! Total time: {total_time / 60:.1f} minutes')

    return model, tokenizer


import math  # For cosine schedule

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MiniGPT')
    parser.add_argument('--config', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size configuration')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per step')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')

    args = parser.parse_args()

    model, tokenizer = train(
        config_name=args.config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        grad_accum_steps=args.grad_accum,
        save_dir=args.save_dir,
    )

    # Generate final sample
    print('Final generation:')
    prompt = 'First Citizen: '
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids], device=next(model.parameters()).device)
    generated = model.generate(input_ids, max_new_tokens=200, temperature=0.7, top_k=40)
    print(tokenizer.decode(generated[0].tolist()))
