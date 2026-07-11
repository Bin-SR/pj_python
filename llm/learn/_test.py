import torch
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# max_len = 5000
# d_model = 512
# position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
# x = position * div_term

# print(position.shape)
# print(div_term.shape)
# print(x.shape)

# xs = torch.sin(x)
# print(xs.shape)

corpus = '''
    the cat sat on the mat
    the dog sat on the log
    the cat and the dog
    the mat and the log
    cat dog mat log
    ''' * 10  # Repeat for more data

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'  # Beginning of Sequence
EOS_TOKEN = '<eos>'  # End of Sequence
# Vocabulary: token -> id， 类型注释，定义一个字典
vocab: Dict[str, int] = {}
        # Reverse vocabulary: id -> token
id_to_token: Dict[int, str] = {}
        # BPE merge rules: (token_a, token_b) -> merged_token
merges: Dict[Tuple[str, str], str] = {}
specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

def _add_token(token):
    if token not in vocab:
        idx = len(vocab)     # 初始化时，0 1 2 3 四个speicla tokens
        vocab[token] = idx   
        id_to_token[idx] = token
        return idx
    return vocab[token]

for token in specials:
    _add_token(token)

chars = sorted(set(corpus))  # 获取text的字符，包括字母和空格和换行，并去除重复字符，再对字符进行排序
for char in chars:
    _add_token(char) 

words = corpus.split()
splits = {word: list(word) for word in set(words)}

print("set: ", set(words))
print(splits)
    
def _word_freqs(words):
    freqs = defaultdict(int)
    for word in words:
        freqs[word] += 1
    return freqs
    
fre = _word_freqs(words)
print(_word_freqs(words), _word_freqs(words).items())
for word, fre in fre.items():
    print(word, fre[word], splits[word])
    