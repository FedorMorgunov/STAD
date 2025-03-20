import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim, max_period=10000):
   """Generate sinusoidal timestep embeddings."""
   half = dim // 2
   freqs = torch.exp(
       -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
   )
   args = timesteps[:, None].float() * freqs[None]
   embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
   if dim % 2:
       embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
   return embedding


def zero_module(module):
   """Zero out the parameters of a module."""
   for p in module.parameters():
       p.detach().zero_()
   return module


class StylizationBlock(nn.Module):
   def __init__(self, latent_dim, time_embed_dim, dropout):
       super().__init__()
       self.emb_layers = nn.Sequential(
           nn.SiLU(),
           nn.Linear(time_embed_dim, 2 * latent_dim)
       )
       self.norm = nn.LayerNorm(latent_dim)
       self.out_layers = nn.Sequential(
           nn.SiLU(),
           nn.Dropout(p=dropout),
           zero_module(nn.Linear(latent_dim, latent_dim))
       )


   def forward(self, h, emb):
       # Compute scale and shift from time embedding
       emb_out = self.emb_layers(emb).unsqueeze(1)  # (B, 1, 2*latent_dim)
       scale, shift = torch.chunk(emb_out, 2, dim=-1)
       h = self.norm(h) * (1 + scale) + shift
       h = self.out_layers(h)
       return h


class FFN(nn.Module):
   def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
       super().__init__()
       self.linear1 = nn.Linear(latent_dim, ffn_dim)
       self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
       self.activation = nn.GELU()
       self.dropout = nn.Dropout(dropout)
       self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)


   def forward(self, x, emb):
       y = self.linear2(self.dropout(self.activation(self.linear1(x))))
       y = x + self.proj_out(y, emb)
       return y


class TemporalSelfAttention(nn.Module):
   def __init__(self, latent_dim, num_head, dropout):
       super().__init__()
       self.num_head = num_head
       self.norm = nn.LayerNorm(latent_dim)
       self.query = nn.Linear(latent_dim, latent_dim, bias=False)
       self.key = nn.Linear(latent_dim, latent_dim, bias=False)
       self.value = nn.Linear(latent_dim, latent_dim, bias=False)
       self.dropout = nn.Dropout(dropout)


   def forward(self, x):
       B, T, D = x.shape
       H = self.num_head
       # Compute queries, keys, and values and split heads
       q = self.query(self.norm(x)).view(B, T, H, D // H).transpose(1,2)  # (B, H, T, D//H)
       k = self.key(self.norm(x)).view(B, T, H, D // H).transpose(1,2)
       v = self.value(self.norm(x)).view(B, T, H, D // H).transpose(1,2)
       attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // H)
       attn = F.softmax(attn_scores, dim=-1)
       attn = self.dropout(attn)
       out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, T, D)
       return out


class AdvancedTransformerBlock(nn.Module):
   def __init__(self, latent_dim, ffn_dim, num_head, dropout, time_embed_dim):
       super().__init__()
       self.self_attn = TemporalSelfAttention(latent_dim, num_head, dropout)
       self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
       self.norm1 = nn.LayerNorm(latent_dim)
       self.norm2 = nn.LayerNorm(latent_dim)
       self.dropout = nn.Dropout(dropout)


   def forward(self, x, emb):
       # Residual self-attention
       attn_out = self.self_attn(x)
       x = x + self.dropout(attn_out)
       x = self.norm1(x)
       # Feed-forward network with residual connection
       ffn_out = self.ffn(x, emb)
       x = x + self.dropout(ffn_out)
       x = self.norm2(x)
       return x


class AdvancedTransformerEncoder(nn.Module):
   def __init__(self, input_dim, latent_dim, ffn_dim, num_layers, num_head, dropout):
       """
       Args:
           input_dim: dimension of input features (e.g., 128 from graph encoder)
           latent_dim: transformer embedding dimension
           ffn_dim: hidden dimension of the feed-forward network
           num_layers: number of advanced transformer blocks
           num_head: number of attention heads
           dropout: dropout probability
       """
       super().__init__()
       # Project input to latent space
       self.input_proj = nn.Linear(input_dim, latent_dim)
       # Learnable positional embeddings
       self.positional_embedding = nn.Parameter(torch.randn(1, 1000, latent_dim))
       self.layers = nn.ModuleList([
           AdvancedTransformerBlock(latent_dim, ffn_dim, num_head, dropout, time_embed_dim=latent_dim)
           for _ in range(num_layers)
       ])
       #self.output_proj = nn.Linear(latent_dim, input_dim)


   def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, latent_dim)
        pos_emb = self.positional_embedding[:, :T, :]
        x = x + pos_emb
        time_cond = pos_emb.mean(dim=1)
        for layer in self.layers:
            x = layer(x, time_cond)
        return x, time_cond  # Return latent_dim dimension

class TemporalCrossAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        B, T, D = x.shape
        H = self.num_head
        x_norm = self.norm(x)
        enc_out_norm = self.norm(enc_out)
        q = self.query(x_norm).view(B, T, H, D//H).transpose(1, 2)
        k = self.key(enc_out_norm).view(B, -1, H, D//H).transpose(1, 2)
        v = self.value(enc_out_norm).view(B, -1, H, D//H).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D//H)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return out

class AdvancedTransformerDecoderBlock(nn.Module):
    def __init__(self, latent_dim, ffn_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.self_attn = TemporalSelfAttention(latent_dim, num_head, dropout)
        self.cross_attn = TemporalCrossAttention(latent_dim, num_head, dropout)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, emb):
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        cross_attn_out = self.cross_attn(x, enc_out)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)
        ffn_out = self.ffn(x, emb)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        return x

class AdvancedTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, ffn_dim, num_layers, num_head, dropout, time_embed_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            AdvancedTransformerDecoderBlock(latent_dim, ffn_dim, num_head, dropout, time_embed_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, enc_out, emb):
        for layer in self.layers:
            x = layer(x, enc_out, emb)
        x = self.output_proj(x)
        return x