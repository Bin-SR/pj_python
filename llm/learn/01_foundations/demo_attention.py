# -*- coding: utf-8 -*-
'''
01_foundations/demo_attention.py - Attention mechanism comprehensive demo

This script demonstrates:
1. How attention weights reveal token relationships
2. Causal masking for autoregressive generation (GPT style)
3. Visual comparison of self-attention vs causal attention
4. How different heads capture different patterns

Run: python demo_attention.py
'''

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from attention import scaled_dot_product_attention, create_causal_mask
from multi_head_attention import MultiHeadAttention


def demo_attention_weights():
    print("=" * 60)
    print("Demo 1: Visualizing Attention Weights")
    print("=" * 60)

    torch.manual_seed(42)
    batch, heads, seq, d_k = 1, 2, 8, 16

    Q = torch.randn(batch, heads, seq, d_k)
    K = torch.randn(batch, heads, seq, d_k)
    V = torch.randn(batch, heads, seq, d_k)

    # No mask: each position attends to all positions
    _, weights_full = scaled_dot_product_attention(Q, K, V)
    print("Without mask (each position sees ALL positions):")
    print(f"  Position 0 attends to: {weights_full[0, 0, 0, :].tolist()}")

    # Causal mask: position i can only see positions <= i
    mask = create_causal_mask(seq)
    _, weights_causal = scaled_dot_product_attention(Q, K, V, mask)
    print("With causal mask (position i sees only positions <= i):")
    for i in range(4):
        w = weights_causal[0, 0, i, :i+1]
        print(f"  Position {i} attends to positions 0-{i}: sum={w.sum().item():.4f}")

    print()

def demo_multi_head_parallel():
    print("=" * 60)
    print("Demo 2: Multi-Head Attention in Action")
    print("=" * 60)

    batch, seq, d_model = 2, 12, 64
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch, seq, d_model)

    # Self-attention
    out = mha(x, x, x)
    print(f"Self-Attention:  {x.shape} -> {out.shape}")
    print(f"  Total params: {sum(p.numel() for p in mha.parameters()):,}")
    print(f"  d_k per head: {d_model // num_heads}")
    print()

def demo_cross_attention_vlm_style():
    print("=" * 60)
    print("Demo 3: Cross-Attention (VLM Style)")
    print("=" * 60)
    print("Simulating: Text tokens querying image features")

    batch, text_len, img_len, d_model = 1, 6, 4, 64
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Text tokens (e.g., "What color is the block?")
    text = torch.randn(batch, text_len, d_model)
    # Image features from vision encoder (e.g., 4 patches)
    image = torch.randn(batch, img_len, d_model)

    # Cross-Attention: Q=text, K=image, V=image
    # The text "queries" the image to extract relevant visual information
    fused = mha(text, image, image)
    print(f"  Text tokens:   {text.shape}")
    print(f"  Image features: {image.shape}")
    print(f"  Fused output:   {fused.shape}")
    print()
    print("This is how VLMs let language 'look at' images!")
    print("In VLA, this fused representation is then used to predict actions.")

def demo_vla_insight():
    print("=" * 60)
    print("Demo 4: Attention Perspective for VLA")
    print("=" * 60)
    print("In a VLA model:")
    print("  1. Vision encoder extracts features from camera images")
    print("  2. Text encoder processes the instruction (e.g., 'pick up cup')")
    print("  3. Cross-Attention fuses: text tokens query image features")
    print("  4. Self-Attention captures temporal dependencies in action sequences")
    print("  5. Action head maps final hidden states to robot commands")
    print()
    print("The Attention mechanism you just learned is the FOUNDATION")
    print("for every step of this pipeline!")

if __name__ == "__main__":
    demo_attention_weights()
    demo_multi_head_parallel()
    demo_cross_attention_vlm_style()
    demo_vla_insight()
    print("All demos complete!")
