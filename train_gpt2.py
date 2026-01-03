from __future__ import annotations
import tiktoken

import math
from dataclasses import dataclass
import torch.nn.functional as F

import torch
from torch import nn
from transformers import GPT2LMHeadModel


class CausalAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Causal = each token can only attend to previous tokens (not future ones).
    This is what makes GPT autoregressive - it predicts next token based on past only.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # n_embd must be divisible by n_head so we can split evenly
        # e.g., 384 / 6 = 64 dimensions per head
        assert config.n_embd % config.n_head == 0

        # Projects input to Q, K, V all at once (3x the size)
        # (B, T, C) -> (B, T, 3*C)
        # e.g., (32, 256, 384) -> (32, 256, 1152)
        #
        # NOTE: Linear layers only operate on the LAST dimension (C).
        # T (256) is NOT a parameter - PyTorch broadcasts the same weights
        # across all 256 positions. Each position is transformed independently.
        # Position interaction happens ONLY in attention (the T×T matrix).
        # Position info comes from: (1) wpe embeddings, (2) attention patterns.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection: combines all heads back to n_embd
        # (B, T, C) -> (B, T, C)
        #
        # Why c_proj when we have MLP next?
        # After concat, heads are just stacked [h1|h2|h3|h4|h5|h6], not mixed.
        # c_proj is a learned (384×384) matrix that MIXES info across all heads.
        # Each output dim can pull from any head's features.
        # MLP is different: (1) has non-linearity, (2) expands/contracts,
        # (3) still processes each position independently (no cross-head mixing).
        # Also: c_proj output goes through residual ADD before MLP sees it.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Mark this layer for scaled initialization (see _init_weights)
        # This is the "exit ramp" onto the residual stream - needs smaller init
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head  # H = 6 heads
        self.n_embd = config.n_embd  # C = 384 embedding dim

        # Causal mask: lower triangular matrix of ones
        # Prevents attending to future tokens
        # Shape: (1, 1, block_size, block_size) for broadcasting
        # e.g., (1, 1, 256, 256)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # x shape: (B, T, C) = (batch, sequence_length, embedding_dim)
        # e.g., (32, 256, 384)
        B, T, C = x.size()

        # Project to Q, K, V combined
        # (B, T, C) -> (B, T, 3*C)
        qkv = self.c_attn(x)

        # Split into Q, K, V each of shape (B, T, C)
        # e.g., (32, 256, 384) each
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention:
        # (B, T, C) -> (B, T, H, C/H) -> (B, H, T, C/H)
        # e.g., (32, 256, 384) -> (32, 256, 6, 64) -> (32, 6, 256, 64)
        # Now each head has its own 64-dim subspace
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention scores: Q @ K^T / sqrt(d_k)
        # (B, H, T, 64) @ (B, H, 64, T) -> (B, H, T, T)
        # e.g., (32, 6, 256, 64) @ (32, 6, 64, 256) -> (32, 6, 256, 256)
        #
        # THIS is where T×T interaction happens!
        # The (256, 256) matrix = each token attending to every other token.
        # This is the ONLY place in the model where positions "talk" to each other.
        # Linear/MLP layers process each position independently (no T×T).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask: set future positions to -inf
        # After softmax, -inf becomes 0 (no attention to future)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over last dim (keys) to get attention weights
        # Each row sums to 1.0
        att = F.softmax(att, dim=-1)

        # Apply attention to values
        # (B, H, T, T) @ (B, H, T, 64) -> (B, H, T, 64)
        # e.g., (32, 6, 256, 256) @ (32, 6, 256, 64) -> (32, 6, 256, 64)
        y = att @ v

        # Reshape back: concatenate all heads
        # Step by step:
        #   y starts as:        (B, H, T, C/H) = (32, 6, 256, 64)
        #   .transpose(1, 2):   (B, T, H, C/H) = (32, 256, 6, 64)  swap H and T dims
        #   .contiguous():      (32, 256, 6, 64)  make memory layout contiguous
        #                       (required before .view() after transpose)
        #   .view(B, T, C):     (32, 256, 384)  flatten last 2 dims: 6*64=384
        #                       = concatenate all 6 heads back together
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        # (B, T, C) -> (B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    """
    Feed-forward network (FFN) in transformer block.

    Two linear layers with GELU activation in between.
    Expands to 4x the embedding dim, then projects back down.
    This gives the model more capacity to learn complex transformations.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Why expand then contract (384 -> 1536 -> 384)?
        #
        # 1. MORE EXPRESSIVE POWER: The 4x expansion creates a larger hidden space
        #    where complex patterns can be detected. Think of it as having more
        #    "neurons" to detect different features before summarizing.
        #
        # 2. NON-LINEAR FEATURE DETECTION: The GELU activation between the two
        #    linear layers allows learning non-linear combinations. Without the
        #    expansion, we'd have less capacity for non-linear transformations.
        #
        # 3. BOTTLENECK ARCHITECTURE: Contract back to 384 to:
        #    - Keep residual stream dimension consistent
        #    - Force the network to compress/summarize what it learned
        #    - Reduce parameter count vs staying at 1536
        #
        # 4. WHY 4x? Empirically works well. Some models use 8/3x (LLaMA) or other
        #    ratios. It's a tradeoff between capacity and compute.

        # Expansion layer: C -> 4*C
        # (B, T, 384) -> (B, T, 1536)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU activation (smoother than ReLU)
        # tanh approximation is faster and used in original GPT-2
        self.gelu = nn.GELU(approximate="tanh")

        # Projection layer: 4*C -> C
        # (B, T, 1536) -> (B, T, 384)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Mark this layer for scaled initialization (see _init_weights)
        # This is the "exit ramp" onto the residual stream - needs smaller init
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # x: (B, T, C) e.g., (32, 256, 384)

        x = self.c_fc(x)  # (32, 256, 384) -> (32, 256, 1536)  expand
        x = self.gelu(x)  # (32, 256, 1536) -> (32, 256, 1536) activation
        x = self.c_proj(x)  # (32, 256, 1536) -> (32, 256, 384)  contract

        return x


class Block(nn.Module):
    """
    Single transformer block: LayerNorm -> Attention -> LayerNorm -> MLP

    Uses pre-norm architecture (LayerNorm before attention/MLP, not after).
    Residual connections add the input back after each sub-layer.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Layer norms normalize across the embedding dimension (C)
        self.ln_1 = nn.LayerNorm(config.n_embd)  # before attention
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  # before MLP
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (B, T, C) e.g., (32, 256, 384)

        # Attention with residual connection
        # x = x + attn(ln_1(x))
        # Residual lets gradients flow directly through the network
        x = x + self.attn(self.ln_1(x))  # (32, 256, 384) -> (32, 256, 384)

        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))  # (32, 256, 384) -> (32, 256, 384)

        return x


@dataclass
class GPTConfig:
    """
    Configuration for GPT model.

    Default values are for a small model (good for learning/debugging).
    GPT-2 small: block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768
    """

    block_size: int = 1024  # T: max sequence length (context window)
    vocab_size: int = 50257  # V: number of unique tokens (e.g., characters) -> 256 byte tokens + 50000 BPE merges + 1 <|endoftext|> token
    n_layer: int = 12  # L: number of transformer blocks
    n_head: int = 12  # H: number of attention heads
    n_embd: int = 768  # C: embedding dimension (must be divisible by n_head)
    # head_dim = n_embd / n_head = 384 / 6 = 64


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) Language Model.

    Architecture:
    1. Token + Position Embeddings
    2. N Transformer Blocks (attention + MLP with residuals)
    3. Final LayerNorm
    4. Output projection to vocabulary logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Token embedding: maps token IDs to vectors
                # (B, T) -> (B, T, C)
                # e.g., (32, 256) -> (32, 256, 384)
                # Lookup table of shape (vocab_size, n_embd) = (65, 384)
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Position embedding: encodes position information
                # Lookup table of shape (block_size, n_embd) = (256, 384)
                # Position 0 gets one vector, position 1 gets another, etc.
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # Stack of transformer blocks
                # Each block: (B, T, C) -> (B, T, C)
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final layer norm before output projection
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # Output projection: embedding dim -> vocab size
        # (B, T, C) -> (B, T, V)
        # e.g., (32, 256, 384) -> (32, 256, 65)
        # Produces logits (unnormalized scores) for each token in vocab
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # GPT-2 paper initialization: scale down weights of residual projections.
                # Each transformer block adds to the residual stream twice (attn.c_proj + mlp.c_proj),
                # so we scale by 1/sqrt(2*n_layer) to keep variance stable as depth increases.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"cannot forward more than {self.config.block_size} tokens"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ---------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("data/tinyshakespeare/input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0

        return x, y


import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
# device = "cpu"  # OVERRRIDE

torch.manual_seed(1337)
torch.mps.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=512 * 2)  # works super-well on M3 with bf16 dtype
# train_loader = DataLoaderLite(B=4, T=1024)
# train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x, y)

optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimiser.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimiser.step()
    torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

print(logits.shape)
print(loss)

import sys

sys.exit(0)

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
print("ok!")

model.eval()
model.to("mps")

# prefix tokes

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to("mps")

torch.manual_seed(42)
torch.mps.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, 1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
