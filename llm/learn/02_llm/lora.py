# -*- coding: utf-8 -*-
'''
02_llm/lora.py - LoRA (Low-Rank Adaptation) for Efficient Fine-tuning

LoRA freezes the base model and injects trainable low-rank matrices into
attention layers. This reduces trainable parameters by >99%, making
fine-tuning possible on consumer GPUs.

Key insight: weight updates during fine-tuning have low "intrinsic rank",
so we can represent update = B @ A where A and B are small matrices.

Standard:   W' = W + delta_W          (train all of W, huge memory)
LoRA:       W' = W + B @ A            (train only B and A, tiny memory)
             where B: (d, r), A: (r, k), r << d

For RTX 3050 (6GB), LoRA is ESSENTIAL for fine-tuning.
You can fine-tune a model that would otherwise need 24GB+ VRAM.

VLA connection: LoRA is used to adapt VLM backbones to robotics tasks
without full retraining. You can take a pretrained VLM and LoRA-fine-tune
it for specific robot manipulation tasks.
'''

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    '''
    LoRA-adapted Linear layer.

    Forward: y = W @ x + (B @ A) @ x * (alpha / r)
             where B: (out_features, r), A: (r, in_features)

    The scaling factor alpha/r controls the magnitude of the LoRA update.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,          # Rank (small = fewer params, 4-16 typical)
        lora_alpha: float = 16.0,  # Scaling factor
        dropout_p: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        # Original (frozen) weights
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA parameters (trainable)
        # A: (r, in_features) - initialized with Kaiming uniform
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        # B: (out_features, r) - initialized with zeros
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Scaling
        self.scaling = lora_alpha / r
        self.r = r

        # Dropout
        self.lora_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Initialize
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        '''Initialize A with Kaiming, B with zeros (so initially LoRA adds 0).'''
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward: base_output + lora_output * scaling

        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        '''
        # Base forward (frozen)
        base_output = self.linear(x)

        # LoRA forward: x -> A -> dropout -> B -> scale
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    @property
    def trainable_params(self) -> int:
        '''Number of trainable parameters in this layer.'''
        return self.lora_A.numel() + self.lora_B.numel()

    @property
    def total_params(self) -> int:
        '''Total parameters (frozen + trainable).'''
        return sum(p.numel() for p in self.parameters())

    @property
    def efficiency_ratio(self) -> float:
        '''Trainable / Total parameter ratio.'''
        return self.trainable_params / self.total_params


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    target_modules: list = None,
) -> nn.Module:
    '''
    Apply LoRA to all attention projection layers in a model.

    This replaces nn.Linear layers with LoRALinear in attention modules.
    Only Q, K, V, O projections are typically targeted (not FFN or embeddings).

    Args:
        model: The base model to adapt
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        target_modules: List of module name patterns to target
                       (default: ['W_Q', 'W_K', 'W_V', 'W_O'])

    Returns:
        Model with LoRA applied (in-place modification)
    '''
    if target_modules is None:
        target_modules = ['W_Q', 'W_K', 'W_V', 'W_O', 'q_proj', 'k_proj', 'v_proj', 'o_proj']

    replaced_count = 0
    for name, module in model.named_modules():
        # Check if this module should be LoRA'd
        should_replace = any(target in name for target in target_modules)

        if should_replace and isinstance(module, nn.Linear):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            # Create LoRA version
            lora_linear = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                bias=module.bias is not None,
            )

            # Copy original weights
            lora_linear.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_linear.linear.bias.data.copy_(module.bias.data)

            # Replace
            setattr(parent, child_name, lora_linear)
            replaced_count += 1

    print(f'LoRA applied: {replaced_count} layers replaced (r={r}, alpha={lora_alpha})')

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)')
    print(f'  Memory saving: {(total - trainable) * 4 / 1e6:.1f} MB (FP32)')

    return model


# ============================================================
# Demo
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('LoRA (Low-Rank Adaptation) Demo')
    print('=' * 60)

    # Test single LoRA layer
    batch, in_dim, out_dim = 4, 256, 512
    lora = LoRALinear(in_dim, out_dim, r=8, lora_alpha=16.0)
    x = torch.randn(batch, in_dim)
    y = lora(x)
    print(f'LoRA Linear: {x.shape} -> {y.shape}')
    print(f'  Trainable: {lora.trainable_params:,}')
    print(f'  Total:     {lora.total_params:,}')
    print(f'  Ratio:     {lora.efficiency_ratio:.4f} ({100*lora.efficiency_ratio:.2f}%)')

    # Demonstrate parameter savings
    print(f'Without LoRA:  Train {out_dim * in_dim + out_dim:,} params')
    print(f'With LoRA(r=8): Train {8 * (in_dim + out_dim):,} params')
    print(f'Savings: {100 * (1 - (8*(in_dim+out_dim))/(out_dim*in_dim+out_dim)):.1f}% fewer trainable params')

    # Show weight initialization
    print(f'lora_A norm: {lora.lora_A.norm().item():.4f} (initialized)')
    print(f'lora_B norm: {lora.lora_B.norm().item():.4f} (zeros, so LoRA adds 0 at start)')
    print('Done!')
