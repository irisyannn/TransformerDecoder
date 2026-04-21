from __future__ import annotations
from flax import nnx
from jaxtyping import Float, Array, Bool, Int
import jax.numpy as jnp
import jax
from einops import rearrange, einsum


class Attention(nnx.Module):
    def __init__(self, num_heads: int, hidden_size: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.Wq = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wk = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wv = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wo = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(
        self,
        query: Float[Array, "B Lq D"],
        key: Float[Array, "B Lk D"],
        value: Float[Array, "B Lk D"],
        mask: Bool[Array, "B 1 Lq Lk"],
    ) -> Float[Array, "B Lq D"]:
        B, Lq, D = query.shape
        _, Lk, _ = key.shape

        Q: Float[Array, "B H Lq d_k"] = rearrange(
            self.Wq(query), "B Lq (H d_k) -> B H Lq d_k"
        )
        K: Float[Array, "B H Lk d_k"] = rearrange(
            self.Wk(key), "B Lk (H d_k) -> B H Lk d_k"
        )
        V: Float[Array, "B H Lk d_k"] = rearrange(
            self.Wv(value), "B Lk (H d_k) -> B H Lk d_k"
        )

        scores: Float[Array, "B H Lq Lk"] = einsum(
            Q, K, "B H Lq d_k, B H Lk d_k -> B H Lq Lk"
        ) / jnp.sqrt(self.head_dim)
        softmax: Float[Array, "B H Lq Lk"] = jax.nn.softmax(scores, axis=-1, where=mask)
        attention: Float[Array, "B H Lq d_k"] = einsum(
            softmax, V, "B H Lq Lk, B H Lk d_k -> B H Lq d_k"
        )
        multi_attention: Float[Array, "B Lq D"] = rearrange(
            attention, "B H Lq d_k -> B Lq (H d_k)"
        )
        return self.Wo(multi_attention)


class Embedding(nnx.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, seq_length: int, rngs: nnx.Rngs
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.Embedding: nnx.Embed = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
        self.Positional: nnx.Embed = nnx.Embed(seq_length, hidden_size, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, vocab_size, rngs=rngs)

    def embedding(self, tokens: Int[Array, "B Lq"]) -> Float[Array, "B Lq D"]:
        curr_seqlength = tokens.shape[1]
        positions: Int[Array, "B Lq"] = jnp.arange(curr_seqlength)[None, :]
        return self.Embedding(tokens) + self.Positional(positions)

    def logits(self, hidden: Float[Array, "B Lq D"]) -> Float[Array, "B Lq V"]:
        return self.linear(hidden)


class Transformer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        interm_size: int,
        num_heads: int,
        dropout_prob: float,
        rngs: nnx.Rngs,
    ):
        self.self_attention = Attention(num_heads, hidden_size, rngs)
        self.cross_attention = Attention(num_heads, hidden_size, rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(hidden_size, interm_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(interm_size, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_prob, rngs=rngs),
        )
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B Lq D"],
        encoder_output: Float[Array, "B Lk D"],
        self_mask: Bool[Array, "B 1 Lq Lk"],
        cross_mask: Bool[Array, "B 1 Lq Lk"],
    ) -> Float[Array, "B Lq D"]:
        norm: Float[Array, "B L D"] = self.norm1(x)
        x += self.self_attention(norm, norm, norm, self_mask)

        norm = self.norm2(x)
        x += self.cross_attention(norm, encoder_output, encoder_output, cross_mask)

        norm = self.norm3(x)
        x += self.ffn(norm)
        return x


class DecoderModel(nnx.Module):
    def __init__(
        self,
        hidden_size,
        interm_size: int,
        num_heads: int,
        vocab_size: int,
        seq_length: int,
        layers: int,
        dropout_prob: float,
        rngs: nnx.Rngs,
    ):
        self.embedding = Embedding(hidden_size, vocab_size, seq_length, rngs)
        self.decoder = nnx.List(
            [
                Transformer(hidden_size, interm_size, num_heads, dropout_prob, rngs)
                for _ in range(layers)
            ]
        )

    def __call__(
        self,
        tokens: Int[Array, "B Lq V"],
        encoder_output: Float[Array, "B Lk D"],
        self_mask: Bool[Array, "B 1 Lq Lk"],
        cross_mask: Bool[Array, "B 1 Lq Lk"],
    ):
        hidden_output: Float[Array, "B Lq D"] = self.embedding.embedding(tokens)
        for layer in self.decoder:
            hidden_output: Float[Array, "B Lq D"] = layer(
                hidden_output, encoder_output, self_mask, cross_mask
            )
        return self.embedding.logits(hidden_output)
