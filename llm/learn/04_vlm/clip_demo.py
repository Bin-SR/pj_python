# -*- coding: utf-8 -*-
"""
04_vlm/clip_demo.py - CLIP-style Image-Text Matching

CLIP learns to match images with text via contrastive learning.
VLA connection: CLIP provides the visual backbone for VLA perception.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleImageEncoder(nn.Module):
    """Simplified CNN image encoder (replaces ViT for learning)."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, dim=-1)


class SimpleTextEncoder(nn.Module):
    """Simple Transformer text encoder."""
    def __init__(self, vocab_size=512, embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Parameter(torch.randn(1, 128, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.positional[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.projection(x)
        return F.normalize(x, dim=-1)


class SimpleCLIP(nn.Module):
    """Simplified CLIP for learning."""
    def __init__(self, embed_dim=512, vocab_size=512):
        super().__init__()
        self.image_encoder = SimpleImageEncoder(embed_dim)
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, images, text_ids):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(text_ids)
        logit_scale = self.logit_scale.exp()
        return logit_scale * img_feat @ txt_feat.T


def contrastive_loss(logits):
    batch = logits.size(0)
    labels = torch.arange(batch, device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


if __name__ == '__main__':
    print('=' * 60)
    print('SimpleCLIP Demo')
    print('=' * 60)
    model = SimpleCLIP(embed_dim=256, vocab_size=512)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    batch = 4
    images = torch.randn(batch, 3, 64, 64)
    text_ids = torch.randint(0, 512, (batch, 32))
    logits = model(images, text_ids)
    loss = contrastive_loss(logits)
    print(f'Logits: {logits.shape}, Loss: {loss.item():.4f}')
    diag = logits.diag().mean().item()
    off = (logits.sum() - logits.diag().sum()) / (batch * (batch - 1))
    print(f'Matching: {diag:.3f}, Non-matching: {off:.3f}')
    print('Done!')