# -*- coding: utf-8 -*-
"""
04_vlm/mini_vlm.py - Mini VLM: Vision + Language Model

Combines vision encoder with MiniGPT to create a simple VLM.
Architecture: Image -> Vision Encoder -> Projection -> [IMG tokens] + [TEXT tokens] -> LLM -> Answer

This mirrors LLaVA, Qwen-VL, GPT-4V at a small scale.
VLA connection: The VLM output hidden states are fed to an Action Head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_llm'))
from mini_gpt import MiniGPT, MiniGPTConfig


class MiniVLM(nn.Module):
    """Mini Vision-Language Model. Maps images + text -> text."""

    def __init__(self, llm_config, vision_embed_dim=384, llm_embed_dim=256):
        super().__init__()
        self.llm = MiniGPT(llm_config)

        # Vision encoder: simple CNN for RTX 3050
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),  # 7x7 = 49 patches
        )
        # Project vision features to LLM embedding space
        self.vision_projection = nn.Sequential(
            nn.Flatten(2),  # (B, 256, 7, 7) -> (B, 256, 49)
            nn.Linear(49, 16),  # 16 visual tokens
            nn.Linear(256, llm_embed_dim),
        )

    def encode_images(self, images):
        features = self.vision_encoder(images)  # (B, 256, 7, 7)
        features = features.flatten(2).transpose(1, 2)  # (B, 49, 256)
        B, N, C = features.shape
        features = features.transpose(1, 2).reshape(B, C, N)  # (B, 256, 49)
        features = self.vision_projection[0](features)  # flatten
        features = self.vision_projection[1](features)  # 49 -> 16
        # features: (B, C, 16) -> (B, 16, C)
        features = features.transpose(1, 2)  # (B, 16, 256)
        features = self.vision_projection[2](features)  # (B, 16, llm_dim)
        return features

    def forward(self, images, text_ids):
        img_tokens = self.encode_images(images)  # (B, 16, llm_dim)
        txt_tokens = self.llm.token_embedding(text_ids)  # (B, T, llm_dim)
        # Concatenate: [IMG tokens | TEXT tokens]
        combined = torch.cat([img_tokens, txt_tokens], dim=1)
        combined = self.llm.position_embedding(combined)
        combined = self.llm.embed_dropout(combined)
        # Causal mask
        seq_len = combined.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=combined.device) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        for block in self.llm.blocks:
            combined = block(combined, mask=mask)
        combined = self.llm.final_norm(combined)
        logits = self.llm.lm_head(combined)
        return logits


if __name__ == '__main__':
    print('=' * 60)
    print('MiniVLM Demo')
    print('=' * 60)
    config = MiniGPTConfig(vocab_size=512, d_model=256, num_layers=4, num_heads=8, d_ff=1024, max_seq_len=256)
    vlm = MiniVLM(config, vision_embed_dim=256, llm_embed_dim=256)
    params = sum(p.numel() for p in vlm.parameters())
    print(f'Params: {params:,} (~{params/1e6:.1f}M)')
    imgs = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 512, (2, 16))
    logits = vlm(imgs, text)
    print(f'Images: {imgs.shape} + Text: {text.shape} -> Logits: {logits.shape}')
    print('Done!')