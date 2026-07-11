# -*- coding: utf-8 -*-
'''
02_llm/tokenizer.py - BPE (Byte-Pair Encoding) Tokenizer

A tokenizer converts raw text into token IDs that the model can process.
BPE is the most common algorithm, used by GPT-2/3/4, LLaMA, etc.

Algorithm:
  1. Start with characters as base vocabulary
  2. Count pair frequencies in the training corpus
  3. Merge the most frequent pair into a new token
  4. Repeat until desired vocabulary size

Why tokenization matters for VLA:
  - Robot instructions need to be tokenized into the model's vocabulary
  - Special tokens mark action boundaries and observation segments
'''

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json
import os


# ============================================================
# BPE Tokenizer Implementation
# ============================================================
class BPETokenizer:
    '''
    A minimal but functional BPE tokenizer.

    Capable of:
    - Training on a text corpus to learn merge rules
    - Encoding text to token IDs
    - Decoding token IDs back to text
    - Saving/loading the vocabulary and merges
    '''

    def __init__(self):
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.BOS_TOKEN = '<bos>'  # Beginning of Sequence
        self.EOS_TOKEN = '<eos>'  # End of Sequence

        # Vocabulary: token -> id， 类型注释，定义一个字典
        self.vocab: Dict[str, int] = {}
        # Reverse vocabulary: id -> token
        self.id_to_token: Dict[int, str] = {}
        # BPE merge rules: (token_a, token_b) -> merged_token
        self.merges: Dict[Tuple[str, str], str] = {}

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        '''Register special tokens in the vocabulary.'''
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in specials:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        '''Add a single token to the vocabulary. Returns its ID.'''
        if token not in self.vocab:
            idx = len(self.vocab)     # 初始化时，0 1 2 3 四个speical tokens
            self.vocab[token] = idx   
            self.id_to_token[idx] = token
            return idx
        return self.vocab[token]

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train(self, text: str, vocab_size: int = 512, min_freq: int = 2):
        '''
        Train BPE tokenizer on raw text.

        Args:
            text: Training corpus (raw string)
            vocab_size: Target vocabulary size
            min_freq: Minimum frequency for a pair to be merged
        '''
        print(f'Training BPE tokenizer...')
        print(f'  Corpus length: {len(text)} characters')
        print(f'  Target vocab:  {vocab_size} tokens')

        # Step 1: Initialize with character-level tokens
        chars = sorted(set(text))  # 获取text的字符，包括字母和空格和换行，并去除重复字符，再对字符进行排序
        for char in chars:
            self._add_token(char) # 初始化之后，再用_add_token，则索引从4开始

        # Step 2: Split text into character sequences (as list of lists)
        # Each word is a list of character tokens
        # 对整个text进行分词,再由set去除重复单词， list(word) = ['c', 'a', 't']
        words = text.split()
        splits = {word: list(word) for word in set(words)}

        print("set: ", set(words))
        print("splits: ", splits, '\n')

        # Step 3: Iteratively merge frequent pairs
        num_merges = 0
        # 当目标词表大于现有词表时：
        while len(self.vocab) < vocab_size:
            # Count pair frequencies
            pair_counts = defaultdict(int)
            print(self._word_freqs(words).items())
            for word, freq in self._word_freqs(words).items():
                symbols = splits[word]
                # 对于实例文本corpus，如果单词长度小于2，那么跳过本次循环，直接进入下次
                if len(symbols) < MIN_WORD_LENGTH:
                    continue
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])  # ?????为什么要分成两个字母两个字母的
                    pair_counts[pair] += freq
                    # print("pair: ", pair)
                    # print("pair_counts[pair]: ", pair_counts[pair])
                # print()
            print(pair_counts)
            print()
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_freq = pair_counts[best_pair]

            print("best_pair: ", best_pair)
            print("best_freq: ", best_freq, '\n')

            if best_freq < min_freq:
                print(f'  Stopping: best pair frequency ({best_freq}) < min_freq ({min_freq})')
                break

            # Create merged token
            merged_token = best_pair[0] + best_pair[1]
            self._add_token(merged_token)
            self.merges[best_pair] = merged_token
            num_merges += 1
            print("merged_token: ", merged_token)
            print("self.merges: ", self.merges)
            print("self.vocab: ", self.vocab)
            print("self.id_to_token: ", self.id_to_token)
            exit()

            # Update splits: replace the pair with merged token in all words
            for word in splits:
                symbols = splits[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1 and
                        symbols[i] == best_pair[0] and
                        symbols[i + 1] == best_pair[1]):
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

        print(f'  Done! {num_merges} merges, vocab size: {len(self.vocab)}')

    def _word_freqs(self, words: List[str]) -> Dict[str, int]:
        '''Count word frequencies.'''
        freqs = defaultdict(int)
        for word in words:
            freqs[word] += 1
        return freqs

    # ----------------------------------------------------------
    # Encoding / Decoding
    # ----------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        '''
        Convert text to token IDs using learned BPE merges.

        Args:
            text: Input text string
            add_special_tokens: If True, prepend BOS and append EOS

        Returns:
            List of token IDs
        '''
        # Split into characters first
        words = text.split()
        tokens = []
        for word in words:
            symbols = list(word)
            # Apply BPE merges greedily
            merged = self._apply_merges(symbols)
            tokens.extend(merged)

        # Convert tokens to IDs (use UNK for unknown tokens)
        ids = [self.vocab.get(t, self.vocab[self.UNK_TOKEN]) for t in tokens]

        if add_special_tokens:
            ids = [self.vocab[self.BOS_TOKEN]] + ids + [self.vocab[self.EOS_TOKEN]]

        return ids

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        '''Apply BPE merge rules to a list of symbols.'''
        # Sort merges for deterministic behavior
        merge_list = sorted(self.merges.items(), key=lambda x: len(x[1]), reverse=True)

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(symbols) - 1:
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merges:
                    symbols = symbols[:i] + [self.merges[pair]] + symbols[i + 2:]
                    changed = True
                else:
                    i += 1

        return symbols

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        '''
        Convert token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: If True, remove special tokens from output

        Returns:
            Decoded text string
        '''
        special_ids = {
            self.vocab[t] for t in
            [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
            if t in self.vocab
        }

        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            token = self.id_to_token.get(idx, self.UNK_TOKEN)
            tokens.append(token)

        return ''.join(tokens)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------
    def save(self, path: str):
        '''Save tokenizer to disk.'''
        data = {
            'vocab': self.vocab,
            'merges': {f'{k[0]} {k[1]}': v for k, v in self.merges.items()},
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'Tokenizer saved to {path}')

    def load(self, path: str):
        '''Load tokenizer from disk.'''
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        for k, v in data['merges'].items():
            a, b = k.split(' ')
            self.merges[(a, b)] = v
        print(f'Tokenizer loaded from {path} (vocab size: {len(self.vocab)})')

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get(self.PAD_TOKEN, 0)

    @property
    def bos_token_id(self) -> int:
        return self.vocab.get(self.BOS_TOKEN, 0)

    @property
    def eos_token_id(self) -> int:
        return self.vocab.get(self.EOS_TOKEN, 0)


# ============================================================
# Demo
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('BPE Tokenizer Demo')
    print('=' * 60)

    # Tiny training corpus, 语料库
    corpus = '''
    the cat sat on the mat
    the dog sat on the log
    the cat and the dog
    the mat and the log
    cat dog mat log
    ''' * 10  # Repeat for more data
    
    MIN_WORD_LENGTH = 2

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=100, min_freq=2)

    # Test encoding/decoding
    test_text = 'the cat sat on the mat'
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)

    print(f'Input:    "{test_text}"')
    print(f'Token IDs: {ids}')
    print(f'Decoded:  "{decoded}"')
    print(f'Vocab size: {tokenizer.vocab_size}')
