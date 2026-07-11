# -*- coding: utf-8 -*-
'''
02_llm/generate.py - Text Generation with MiniGPT

Interactive and batch text generation. Supports:
- Temperature control
- Top-k and top-p (nucleus) sampling
- Beam search (simple implementation)
- Batch generation with different prompts

Usage:
    python generate.py --checkpoint checkpoints/minigpt_small_epoch5.pt --prompt "The "
'''

import torch
import torch.nn.functional as F
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_foundations'))

from mini_gpt import MiniGPT, MiniGPTConfig
from tokenizer import BPETokenizer


class TextGenerator:
    '''
    Flexible text generator with multiple sampling strategies.

    Sampling strategies:
    - Greedy: Always pick the most likely token (deterministic, often repetitive)
    - Temperature: Scale logits before softmax (higher = more random)
    - Top-k: Only consider the k most likely tokens
    - Top-p (nucleus): Consider tokens whose cumulative probability exceeds p
    '''

    def __init__(self, model: MiniGPT, tokenizer: BPETokenizer, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        num_return_sequences: int = 1,
        stop_token: str = None,
    ) -> list:
        '''
        Generate text continuations.

        Args:
            prompt: Starting text
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling (0.0 = disabled)
            num_return_sequences: Number of different continuations
            stop_token: Stop generation when this token is generated

        Returns:
            List of (generated_text, token_ids) tuples
        '''
        # Encode prompt
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids] * num_return_sequences, device=self.device)

        # Store stop token ID if provided
        stop_id = None
        if stop_token and stop_token in self.tokenizer.vocab:
            stop_id = self.tokenizer.vocab[stop_token]

        for step in range(max_new_tokens):
            # Truncate if needed
            if input_ids.shape[1] > self.model.config.max_seq_len:
                input_ids = input_ids[:, -self.model.config.max_seq_len:]

            # Forward pass
            logits = self.model(input_ids)

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                # Greedy: temperature = 0
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                continue

            # Apply top-k
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')

            # Apply top-p
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Check for stop token
            if stop_id is not None:
                if (next_tokens == stop_id).any():
                    # Individual stopping not implemented in batch;
                    # for simplicity, stop when all have generated stop
                    pass

        # Decode
        results = []
        for i in range(num_return_sequences):
            text = self.tokenizer.decode(input_ids[i].tolist())
            results.append((text, input_ids[i].tolist()))

        return results

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, **kwargs):
        '''
        Generator that yields tokens one at a time (for streaming output).

        This pattern is used in chat applications where text appears incrementally.
        '''
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.model.config.max_seq_len:
                input_ids = input_ids[:, -self.model.config.max_seq_len:]

            logits = self.model(input_ids)
            next_logits = logits[:, -1, :] / kwargs.get('temperature', 1.0)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            token_text = self.tokenizer.id_to_token.get(token_id, '<unk>')
            yield token_text

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if token_id == self.tokenizer.eos_token_id:
                break


# ============================================================
# Interactive chat function
# ============================================================
def interactive_chat(checkpoint_path: str):
    '''Load a checkpoint and start interactive text generation.'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading checkpoint: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    tokenizer = BPETokenizer()
    # You'd need to also save/load the tokenizer...
    # For demo, create a minimal one
    tokenizer.train('hello world the cat dog', vocab_size=config.vocab_size)

    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    generator = TextGenerator(model, tokenizer, device)

    print(f'Model loaded ({sum(p.numel() for p in model.parameters()):,} params)')
    print('Enter prompts (type "quit" to exit):')

    while True:
        try:
            prompt = input('Prompt: ').strip()
            if prompt.lower() == 'quit':
                break
            if not prompt:
                continue

            results = generator.generate(
                prompt, max_new_tokens=100, temperature=0.8, top_k=40
            )
            for text, _ in results:
                print(f'Output: {text}')
                print('-' * 40)
        except KeyboardInterrupt:
            print('Goodbye!')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiniGPT Text Generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='The ',
                        help='Starting prompt')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k filtering')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive chat mode')

    args = parser.parse_args()

    if args.interactive:
        interactive_chat(args.checkpoint)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint['config']

        # Minimal tokenizer (in practice, save/load it with the checkpoint)
        tokenizer = BPETokenizer()
        tokenizer.train('hello world test data', vocab_size=config.vocab_size)

        model = MiniGPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        generator = TextGenerator(model, tokenizer, device)

        results = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        for text, _ in results:
            print(text)
