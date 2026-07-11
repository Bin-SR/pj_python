# -*- coding: utf-8 -*-
'''
02_llm/kv_cache.py - KV Cache for Fast Inference

When generating text autoregressively, we recompute attention for ALL
previous tokens at each step. This is wasteful! KV Cache stores the
Key and Value tensors from previous steps and reuses them.

Without KV Cache: O(n^2) per generation step (recompute all)
With KV Cache:    O(n) per generation step (only compute new token)

This is CRITICAL for:
- LLM inference (GPT, LLaMA, etc.)
- VLA real-time control (latency matters for robots)
- Any autoregressive generation

Implementation approach:
  1. Modify attention to accept past KV and return updated KV
  2. During generation, pass KV cache between steps
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_foundations'))


class MultiHeadAttentionWithCache(nn.Module):
    '''
    Multi-Head Attention with KV Cache support.

    This is a modified version that can store and reuse past K, V tensors.
    '''

    def __init__(self, d_model=512, num_heads=8, dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, query, key=None, value=None, mask=None,
                past_kv=None, use_cache=False):
        '''
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model) - optional, defaults to query
            value: (batch, seq_len, d_model)
            mask: attention mask
            past_kv: tuple of (past_K, past_V) or None
            use_cache: if True, return updated KV cache

        Returns:
            output: (batch, seq_len, d_model)
            new_kv: tuple of (K, V) if use_cache else None
        '''
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size = query.shape[0]

        # Project
        Q = self._split_heads(self.W_Q(query))
        K = self._split_heads(self.W_K(key))
        V = self._split_heads(self.W_V(value))

        # Append past KV if available
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)  # dim=2 is seq_len
            V = torch.cat([past_V, V], dim=2)

        # Store for cache
        new_kv = (K, V) if use_cache else None

        # Scaled dot-product attention
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Adjust mask for cached length
            if past_kv is not None and mask.size(-1) != K.size(-2):
                mask = mask[:, :, :, -Q.size(2):]  # Take last part
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        output = self._merge_heads(output)
        output = self.W_O(output)

        return output, new_kv


class KVCacheGenerator:
    '''
    Text generator using KV Cache for faster inference.

    Speed comparison (approximate):
    - Without cache: each step is O(L^2) where L is total length
    - With cache:    each step is O(L) for the new token

    For a 100-token generation starting from 10 tokens:
    - Without cache: ~10x slower
    - With cache:    ~1x (only computes new token)
    '''

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=0):
        '''
        Generate text using KV Cache.

        Note: This is a conceptual implementation. The actual MiniGPT
        model would need to be modified to use MultiHeadAttentionWithCache.
        For now, this demonstrates the KV Cache concept.
        '''
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        # In a real implementation, we would:
        # 1. Run the prompt through the model once to populate the cache
        # 2. For each new token, only compute the new token's attention
        #    using the cached K, V from previous tokens

        generated = []
        for _ in range(max_new_tokens):
            # Truncate
            if input_ids.shape[1] > self.model.config.max_seq_len:
                input_ids = input_ids[:, -self.model.config.max_seq_len:]

            # Forward (in real impl, would use cached KV)
            logits = self.model(input_ids)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Decode only the generated tokens
        gen_text = self.tokenizer.decode(generated)
        return prompt + gen_text


# ============================================================
# Performance comparison demo
# ============================================================
def benchmark_kv_cache():
    '''Demonstrate the speed difference with and without KV Cache.'''
    import time

    print('=' * 60)
    print('KV Cache Performance Demo (Conceptual)')
    print('=' * 60)
    print()
    print('In autoregressive generation:')
    print()
    print('Without KV Cache:')
    print('  Step 1: Compute attention for tokens [0]')
    print('  Step 2: Compute attention for tokens [0,1]  <- recomputes [0]')
    print('  Step 3: Compute attention for tokens [0,1,2] <- recomputes [0,1]')
    print('  Total: O(N^3) operations for N generated tokens')
    print()
    print('With KV Cache:')
    print('  Step 1: Compute attention for token [0], cache K,V')
    print('  Step 2: Attention only for token [1], append to cache')
    print('  Step 3: Attention only for token [2], append to cache')
    print('  Total: O(N^2) operations for N generated tokens')
    print()
    print('For 1000 tokens:')
    print(f'  Without cache: ~{1000**3:,} operations (way too slow)')
    print(f'  With cache:    ~{1000**2:,} operations ({1000}x faster!)')
    print()
    print('This is why EVERY production LLM uses KV Cache!')
    print('And why VLA models need it for real-time robot control.')


if __name__ == '__main__':
    benchmark_kv_cache()

    # Test attention with cache
    batch, seq, d_model = 2, 4, 64
    mha = MultiHeadAttentionWithCache(d_model, num_heads=4)

    # First pass: process full sequence
    x = torch.randn(batch, seq, d_model)
    out1, kv = mha(x, use_cache=True)
    print(f'First pass: {x.shape} -> {out1.shape}')
    print(f'  KV cache keys: K={kv[0].shape}, V={kv[1].shape}')

    # Second pass: process new token with cached KV
    new_token = torch.randn(batch, 1, d_model)
    out2, kv2 = mha(new_token, past_kv=kv, use_cache=True)
    print(f'Second pass (with cache): {new_token.shape} -> {out2.shape}')
    print(f'  Updated KV: K={kv2[0].shape}, V={kv2[1].shape}')
    print(f'  Only {new_token.shape[1]} new token computed, rest from cache!')
    print('Done!')
