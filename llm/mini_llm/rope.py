# RoPE : Rotary Position Embedding, 旋转位置编码，使得embedding有了顺序

import torch

class RotaryEmbedding:

    def __init__(self, dim, max_position=512, base=10000):

        self.dim = dim

        theta = 1.0 / ( base ** ( torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_position)
        freqs = torch.outer(position, theta)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.cos = torch.repeat_interleave(cos, 2, dim=-1)
        self.sin = torch.repeat_interleave(sin, 2, dim=-1)


    def rotate_half(self,x):

        x1 = x[...,::2]

        x2 = x[...,1::2]

        return torch.stack((-x2,x1), dim=-1).flatten(-2)


    def apply(self,x):

        seq_len=x.shape[-2]

        cos=self.cos[:seq_len]
        sin=self.sin[:seq_len]

        # 增加batch广播维度
        cos=cos.unsqueeze(0)
        sin=sin.unsqueeze(0)

        return (x*cos + self.rotate_half(x)*sin)
    
Test = 1
if Test == 1:
    x=torch.randn(1, 4, 8)
    rope=RotaryEmbedding(dim=8)
    y=rope.apply(x)

    print(x.shape)
    print(y.shape)