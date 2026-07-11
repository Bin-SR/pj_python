# -*- coding: utf-8 -*-
"""
05_vla/vla_model.py - Full VLA Model Assembly
Assembles VLM + Action Head into complete VLA: Image+Text -> Action
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_llm'))
sys.path.insert(0, os.path.dirname(__file__))
from mini_gpt import MiniGPT, MiniGPTConfig
from action_head import ActionHead, ActionChunkHead, DiscretizedActionHead


class VLAModel(nn.Module):
    """Complete VLA: Vision + Language -> Action."""
    def __init__(self, llm_config, action_dim=7, action_type='continuous', chunk_size=10):
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        self.llm = MiniGPT(llm_config)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.vision_proj = nn.Sequential(
            nn.Flatten(2), nn.Linear(16, 8), nn.Linear(256, llm_config.d_model),
        )
        if action_type == 'continuous':
            self.action_head = ActionHead(llm_config.d_model, action_dim)
        elif action_type == 'chunk':
            self.action_head = ActionChunkHead(llm_config.d_model, action_dim, chunk_size)
        elif action_type == 'discrete':
            self.action_head = DiscretizedActionHead(llm_config.d_model, action_dim)

    def encode_images(self, images):
        f = self.vision_encoder(images).flatten(2).transpose(1, 2)
        B, N, C = f.shape
        f = f.transpose(1, 2).reshape(B, C, N)
        f = self.vision_proj[0](f)
        f = self.vision_proj[1](f)
        f = f.transpose(1, 2)
        return self.vision_proj[2](f)

    def forward(self, images, text_ids):
        img_tok = self.encode_images(images)
        txt_tok = self.llm.token_embedding(text_ids)
        x = torch.cat([img_tok, txt_tok], dim=1)
        x = self.llm.position_embedding(x)
        x = self.llm.embed_dropout(x)
        sl = x.shape[1]
        mask = torch.triu(torch.ones(sl, sl, device=x.device) * float('-inf'), diagonal=1).unsqueeze(0).unsqueeze(0)
        for block in self.llm.blocks:
            x = block(x, mask=mask)
        x = self.llm.final_norm(x)
        return self.action_head(x)


if __name__ == '__main__':
    print('=' * 60)
    print('VLA Model Demo')
    print('=' * 60)
    config = MiniGPTConfig(vocab_size=512, d_model=256, num_layers=4, num_heads=8, d_ff=1024, max_seq_len=256)
    vla = VLAModel(config, action_dim=7, action_type='continuous')
    print(f'Params: {sum(p.numel() for p in vla.parameters()):,}')
    imgs = torch.randn(2, 3, 224, 224)
    txt = torch.randint(0, 512, (2, 16))
    actions = vla(imgs, txt)
    print(f'Input: {imgs.shape} + {txt.shape} -> Action: {actions.shape}')
    print('Done!')