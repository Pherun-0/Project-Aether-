import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Llama/Mistral)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.head_dim * num_kv_heads
        
        # Attention scaling factor
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(0.1)
        self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA: Repeat KV heads
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Apply RoPE
        cos, sin = self.rope(T, x.device)
        cos = cos[None, None, :, :]  # (1, 1, T, head_dim)
        sin = sin[None, None, :, :]
        
        def rotate_half(tensor):
            half = tensor.shape[-1] // 2
            return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
        
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Output
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.w_up = nn.Linear(embed_dim, ffn_dim * 2, bias=False)  # For SwiGLU split
        self.w_down = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.w_up(x)
        x = SwiGLU()(x_up)
        x = self.dropout(x)
        return self.w_down(x)

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attn(self.attn_norm(x), mask)
        x = x + attn_output
        
        # Feed-forward with residual
        ffn_output = self.ffn(self.ffn_norm(x))
        x = x + ffn_output
        return x

class LMHead(nn.Module):
    """Tied LM Head that reuses embedding weights to save params."""
    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute logits using transposed embedding weights (no extra params)
        return x @ self.embed_tokens.weight.t()

class ReasoningTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 812,
        ffn_dim: int = 2358,
        num_layers: int = 24,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, ffn_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)
        
        # Tied LM head (reuses embed_tokens weights)
        self.lm_head = LMHead(self.embed_tokens)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        # Token embeddings
        x = self.embed_tokens(input_ids)
        
        # Causal mask
        if attention_mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        else:
            mask = attention_mask[:, None, None, :].float()
        
        # Layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Loss if provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return logits, loss

# Test the model
if __name__ == "__main__":
    model = ReasoningTransformer()
    input_ids = torch.randint(0, 32000, (2, 512))
    logits, loss = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,} (~{sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    print("✅ Model forward pass successful!")