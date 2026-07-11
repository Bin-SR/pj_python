# -*- coding: utf-8 -*-
"""
05_vla/train_vla.py - VLA Training Script

Trains VLA model using demonstration data.
Flow: collect demos -> tokenize -> train -> evaluate
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_llm'))
from vla_model import VLAModel
from mini_gpt import MiniGPTConfig
from tokenizer import BPETokenizer


class VLADataset(Dataset):
    """Dataset: (image, instruction, action) triples."""
    def __init__(self, images, instructions, actions, tokenizer, max_text_len=32):
        self.images = images
        self.actions = actions
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.instruction_ids = []
        for inst in instructions:
            ids = tokenizer.encode(inst, add_special_tokens=True)
            ids = ids[:max_text_len]
            while len(ids) < max_text_len:
                ids.append(tokenizer.pad_token_id)
            self.instruction_ids.append(ids)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        txt = torch.tensor(self.instruction_ids[idx], dtype=torch.long)
        act = torch.tensor(self.actions[idx], dtype=torch.float32)
        return img, txt, act


def generate_synthetic_data(num_samples=200):
    """Generate synthetic VLA training data."""
    images = torch.randn(num_samples, 3, 224, 224)
    instructions = []
    actions = []
    for i in range(num_samples):
        x = i / num_samples
        if x < 0.33:
            instructions.append('pick up the red block')
            actions.append([0.5 + x, 0.3, 0.2 + x, 0.0, 0.0, 0.0, 1.0])
        elif x < 0.66:
            instructions.append('place the block on the table')
            actions.append([0.5 - x, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0])
        else:
            instructions.append('move to home position')
            actions.append([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    return images, instructions, actions


def train_vla(epochs=5):
    print('=' * 60)
    print('VLA Training Demo')
    print('=' * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    config = MiniGPTConfig(vocab_size=512, d_model=256, num_layers=4, num_heads=8, d_ff=1024)
    vla = VLAModel(config, action_dim=7, action_type='continuous').to(device)
    tokenizer = BPETokenizer()
    images, instructions, actions = generate_synthetic_data(200)
    tokenizer.train(' '.join(instructions), vocab_size=config.vocab_size)
    dataset = VLADataset(images, instructions, actions, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(vla.parameters(), lr=3e-4)
    print(f'Training on {len(dataset)} samples for {epochs} epochs...')
    for epoch in range(epochs):
        total_loss = 0
        for imgs, txt_ids, acts in loader:
            imgs = imgs.to(device)
            txt_ids = txt_ids.to(device)
            acts = acts.to(device)
            pred = vla(imgs, txt_ids)
            loss = F.mse_loss(pred, acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f'  Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}')
    with torch.no_grad():
        test_img = images[:3].to(device)
        test_ids = torch.tensor(dataset.instruction_ids[:3]).to(device)
        test_pred = vla(test_img, test_ids)
        print('Test predictions:')
        print(test_pred.cpu().numpy())
        print('Test ground truth:')
        print(actions[:3])
    print('Training complete!')


if __name__ == '__main__':
    train_vla(epochs=5)