# -*- coding: utf-8 -*-
# =============================================================================
# VLA 语言理解模块
# 将自然语言指令编码为固定维度特征向量。
#
# 提供两种实现:
#   1. SimpleTextEncoder —— 轻量词嵌入 + 1D 卷积 (默认，零外部依赖)
#   2. 可选接入 sentence-transformers / HuggingFace 模型
# =============================================================================

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config import VOCAB_SIZE, TEXT_EMBED_DIM, LANGUAGE_FEATURE_DIM


# ============================================================
# 简易分词器
# ============================================================

class SimpleTokenizer:
    """简易英文分词器，构建固定大小的词表。

    将文本转为小写、按非字母字符分割，映射到词 ID。
    超出词表的词映射到 <UNK> (id=1)。
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = 32):
        """
        Args:
            vocab_size: 词汇表大小 (含 <PAD>=0, <UNK>=1)
            max_len: 最大序列长度
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        # 特殊 token
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        # 词表: {"word": id}
        self._word2id = {"<PAD>": 0, "<UNK>": 1}
        self._next_id = 2

    def build_vocab(self, texts: list):
        """从文本列表中构建词表。

        Args:
            texts: 字符串列表
        """
        word_freq = {}
        for text in texts:
            for word in self._tokenize_text(text):
                word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序，取 top-K
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        for word, _ in sorted_words:
            if self._next_id >= self.vocab_size:
                break
            if word not in self._word2id:
                self._word2id[word] = self._next_id
                self._next_id += 1

    def _tokenize_text(self, text: str) -> list:
        """将文本分词。"""
        text = text.lower().strip()
        return re.findall(r"[a-zA-Z]+", text)

    def encode(self, text: str) -> torch.Tensor:
        """将文本编码为词 ID 序列。

        Args:
            text: 输入字符串
        Returns:
            (max_len,) LongTensor, 不足补 PAD
        """
        tokens = self._tokenize_text(text)
        ids = []
        for token in tokens[:self.max_len]:
            ids.append(self._word2id.get(token, self.UNK_IDX))
        # 填充
        while len(ids) < self.max_len:
            ids.append(self.PAD_IDX)
        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(self, texts: list) -> torch.Tensor:
        """批量编码。

        Args:
            texts: 字符串列表
        Returns:
            (batch, max_len) LongTensor
        """
        return torch.stack([self.encode(t) for t in texts])


# ============================================================
# 简易文本编码器
# ============================================================

class SimpleTextEncoder(nn.Module):
    """轻量文本编码器: Embedding + 1D Conv + Global Pool。

    参数量极低 (~150K)，适合与视觉编码器联合端到端训练。
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = TEXT_EMBED_DIM,
                 feature_dim: int = LANGUAGE_FEATURE_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 1D 卷积提取局部特征
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # 全局平均池化 + 全连接
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) LongTensor
        Returns:
            (batch, feature_dim) 语言特征
        """
        # (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = self.embedding(token_ids).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 全局平均池化 (忽略 padding)
        mask = (token_ids != 0).float().unsqueeze(1)  # (batch, 1, seq_len)
        x = x * mask
        x = x.sum(dim=2) / (mask.sum(dim=2) + 1e-8)  # (batch, 128)
        x = self.fc(x)
        return x


# ============================================================
# 语言编码器工厂
# ============================================================

# 预设的操作指令模板 (用于构建词表)
DEFAULT_INSTRUCTIONS = [
    "grasp the red cube",
    "pick up the red block",
    "grab the red cube on the table",
    "reach for the red cube",
    "move to the red block and grasp it",
    "pick the red cube",
    "go to the red cube and close the gripper",
    "approach the red block",
    "lift the red cube",
    "put the red cube up",
    "fetch the red block",
    "take the red cube",
    "get the red block from the table",
    "move the arm to grasp the red object",
    "close fingers on the red cube",
]


def create_default_tokenizer() -> SimpleTokenizer:
    """创建并初始化一个包含操作指令词汇的分词器。"""
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(DEFAULT_INSTRUCTIONS)
    return tokenizer


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    tokenizer = create_default_tokenizer()
    print(f"词表大小: {tokenizer.vocab_size}")

    text = "grasp the red cube"
    ids = tokenizer.encode(text)
    print(f"'{text}' -> {ids.tolist()}")

    encoder = SimpleTextEncoder()
    batch = tokenizer.encode_batch(["grasp the red cube", "pick up the block"])
    features = encoder(batch)
    print(f"编码输出: {features.shape}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")
