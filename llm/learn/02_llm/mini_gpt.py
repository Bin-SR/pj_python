# -*- coding: utf-8 -*-
'''
02_llm/mini_gpt.py - MiniGPT: A small GPT model for learning

This is a minimal but complete GPT implementation (~10M parameters),
designed to run on RTX 3050 (6GB VRAM). It covers:

1. Token + Position embeddings
2. Stacked Transformer Blocks (Decoder-only)
3. Causal self-attention
4. LM Head for next-token prediction
5. Weight initialization (GPT-2 style)

Architecture:
    Input IDs -> Embeddings -> [TransformerBlock x N] -> LM Head -> Logits

Usage:
    model = MiniGPT(vocab_size=512, d_model=256, num_layers=6, num_heads=8)
    logits = model(input_ids)  # (batch, seq_len, vocab_size)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Import our custom Transformer components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_foundations'))
from transformer_block import TransformerBlock
from positional_encoding import LearnedPositionalEncoding


class MiniGPTConfig:
    '''Configuration for MiniGPT model. Centralizes all hyperparameters.'''

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout_p: float = 0.1,
        # RTX 3050 friendly defaults
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout_p

        # Validate
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_k = d_model // num_heads

    def __repr__(self):
        params_est = self._estimate_params()
        return (
            f'MiniGPTConfig(vocab={self.vocab_size}, d_model={self.d_model}, '
            f'layers={self.num_layers}, heads={self.num_heads}, '
            f'max_len={self.max_seq_len}, est_params={params_est/1e6:.1f}M)'
        )

    def _estimate_params(self):
        '''Rough parameter count estimate.'''
        # Embeddings
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        # Per layer: 4 x (d_model^2) for attention + 2 x (d_model * d_ff) for FFN
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff
        # LM head
        head = self.d_model * self.vocab_size
        return emb + self.num_layers * per_layer + head


class MiniGPT(nn.Module):
    '''
    MiniGPT - A compact GPT model for educational purposes.

    Key design choices (matching GPT-2):
    - Decoder-only architecture with causal masking
    - Pre-Norm (LayerNorm before Attention/FFN)
    - Learned positional embeddings
    - GELU activation in FFN
    - Weight tying: LM head shares weights with token embeddings (optional)
    '''

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings: convert token IDs to dense vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings: inject position information
        self.position_embedding = LearnedPositionalEncoding(
            d_model=config.d_model, max_len=config.max_seq_len
        )

        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(config.dropout_p)

        # Stack of Transformer blocks (the "depth" of GPT)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout_p=config.dropout_p,
            )
            for _ in range(config.num_layers)
        ])

        # Final LayerNorm (applied after all blocks)
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM Head: projects hidden states to vocabulary logits
        # Weight tying: share with token embedding (reduces params, improves performance)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)
        print(f'MiniGPT initialized: ~{self._count_params()/1e6:.1f}M parameters')

    def _init_weights(self, module):
        '''GPT-2 style weight initialization.'''
        if isinstance(module, nn.Linear):
            # Normal distribution with scaled std for residual projections
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        '''
        Create causal attention mask.

        Returns (1, 1, seq_len, seq_len) mask where upper triangle is -inf.
        '''
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) prediction logits
        '''
        batch_size, seq_len = input_ids.shape

        # 1. Token + Position embeddings
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.position_embedding(token_emb)        # Add position info
        x = self.embed_dropout(x)

        # 2. Create causal mask (once, shared across layers)
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # 3. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # 4. Final normalization
        x = self.final_norm(x)

        # 5. LM Head -> logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        '''
        Autoregressive text generation.

        Args:
            input_ids: (batch, seq_len) starting token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (1.0 = normal, <1 = sharper, >1 = wilder)
            top_k: Keep only top-k tokens (0 = disabled)
            top_p: Nucleus sampling threshold (0 = disabled)

        Returns:
            (batch, seq_len + max_new_tokens) generated sequence
        '''
        self.eval()
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Truncate if sequence exceeds max length
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits = self(input_ids)  # (batch, seq_len, vocab_size)

            # Get logits for the last position only
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if all batches generated EOS
            # (tokenizer-dependent, simplified here)

        return input_ids

    def configure_optimizers(self, learning_rate: float = 3e-4, weight_decay: float = 0.01):
        '''
        Create optimizer with weight decay (AdamW).

        GPT-style: no weight decay on biases and LayerNorm parameters.
        '''
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'layer_norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=learning_rate, betas=(0.9, 0.95))

        return optimizer


# ============================================================
# Config presets for RTX 3050
# ============================================================
def config_tiny() -> MiniGPTConfig:
    '''Tiny config (~4M params) - fast training, minimal VRAM.'''
    return MiniGPTConfig(
        vocab_size=512, d_model=128, num_layers=4,
        num_heads=4, d_ff=512, max_seq_len=256
    )

def config_small() -> MiniGPTConfig:
    '''Small config (~10M params) - good balance for RTX 3050.'''
    return MiniGPTConfig(
        vocab_size=512, d_model=256, num_layers=6,
        num_heads=8, d_ff=1024, max_seq_len=512
    )

def config_medium() -> MiniGPTConfig:
    '''Medium config (~25M params) - RTX 3050 upper limit.'''
    return MiniGPTConfig(
        vocab_size=1024, d_model=384, num_layers=8,
        num_heads=8, d_ff=1536, max_seq_len=512
    )


# ============================================================
# Demo
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('MiniGPT Model Demo')
    print('=' * 60)

    # Create model
    config = config_small()
    print(f'Config: {config}')
    model = MiniGPT(config)

    # Test forward pass
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    print(f'Input:  {input_ids.shape}')
    print(f'Logits: {logits.shape}')
    print(f'  Each position predicts distribution over {config.vocab_size} tokens')

    # Test generation
    print('Generating text...')
    start_ids = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_ids, max_new_tokens=20, temperature=0.8)
    print(f'  Start: {start_ids.shape} -> Generated: {generated.shape}')

    # Memory estimate
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable:    {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print('Done!')
