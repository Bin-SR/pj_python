# -*- coding: utf-8 -*-
"""
04_vlm/vision_encoder.py - Vision Encoder for VLM

Converts images into feature vectors that LLMs can process.
Key components:
  1. Patch embedding: split image into patches, embed each
  2. ViT (Vision Transformer): process patches with self-attention
  3. Projection: map visual features to LLM embedding space

VLA: This is the 'eyes' of the robot. Camera images go through
     the vision encoder to produce tokens the VLA can reason about.
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and embed each as a token."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.projection(x)       # (batch, embed_dim, 14, 14)
        x = x.flatten(2)             # (batch, embed_dim, 196)
        x = x.transpose(1, 2)        # (batch, 196, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) - applies self-attention to image patches.

    Process:
      1. Split image into patches + embed
      2. Add [CLS] token + position embeddings
      3. Pass through Transformer encoder
      4. [CLS] token output = image representation
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio), activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                        # (B, N, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)           # (B, N+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]  # cls_token, patch_tokens


class VisionEncoderForVLM(nn.Module):
    """
    Complete vision encoder for VLM/VLA.
    Outputs visual tokens ready for LLM consumption.
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, llm_dim=512, depth=6, num_heads=12):
        super().__init__()
        self.vit = VisionTransformer(img_size, patch_size, 3, embed_dim, depth, num_heads)
        self.projection = nn.Linear(embed_dim, llm_dim)  # Map to LLM space

    def forward(self, images):
        cls_token, patch_tokens = self.vit(images)
        visual_tokens = self.projection(patch_tokens)  # (B, N, llm_dim)
        return visual_tokens


if __name__ == '__main__':
    print('=' * 60)
    print('Vision Encoder Demo')
    print('=' * 60)
    model = VisionEncoderForVLM(img_size=224, embed_dim=384, llm_dim=256, depth=4, num_heads=6)
    params = sum(p.numel() for p in model.parameters())
    print(f'Params: {params:,} (~{params/1e6:.1f}M)')
    imgs = torch.randn(2, 3, 224, 224)
    tokens = model(imgs)
    print(f'Input: {imgs.shape} -> Visual tokens: {tokens.shape}')
    print(f'Each image -> {tokens.shape[1]} tokens of dim {tokens.shape[2]}')
    print('Done!')