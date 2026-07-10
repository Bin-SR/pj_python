# GQA: Grouped Query Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQAAttention(nn.Module):

    def __init__(self, config):

        super().__init__()


        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim


        self.q_proj = nn.Linear(config.hidden_size, self.num_heads*self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads*self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads*self.head_dim)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, config.hidden_size)
    
    def repeat_kv(self, x):

        B, H, T, D = x.shape

        n_rep = self.num_heads // self.num_kv_heads

        if n_rep == 1:
            return x

        x=x[:, :, None, :, :]
        x=x.expand(B, H, n_rep, T, D)
        x=x.reshape(B, H*n_rep, T, D)

        return x
    
    def forward(self,x):

        B, T, C = x.shape

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        # reshape
        q=q.view(B, T, self.num_heads, self.head_dim)
        k=k.view(B, T, self.num_kv_heads, self.head_dim)
        v=v.view(B, T, self.num_kv_heads, self.head_dim)

        # transpose
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        return q,k,v
    
Test = 1
if Test == 1:
    from config import LlamaConfig
    config=LlamaConfig()
    attn=GQAAttention(config)

    x = torch.randn(2, 8, 256)
    q, k, v = attn(x)

    print(q.shape, k.shape, v.shape)