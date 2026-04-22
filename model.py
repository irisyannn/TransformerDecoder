from __future__ import annotations
from flax import nnx
from jaxtyping import Float, Array, Bool, Int
import jax.numpy as jnp
import jax
from einops import rearrange, einsum
from typing import Optional


class Attention(nnx.Module):
    def __init__(self, num_heads: int, hidden_size: int, rngs: nnx.Rngs):
        self.num_heads: int = num_heads
        self.hidden_size: int = hidden_size
        self.head_dim: int = self.hidden_size // self.num_heads
        self.Wq = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wq = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wk = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wv = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wo = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(
        self,
        query: Float[Array, "B L D"],
        key: Float[Array, "B L D"],
        value: Float[Array, "B L D"],
        mask: Bool[Array, "B 1 L L"],
    ) -> Float[Array, "B L D"]:
        B, L, D = query.shape

        Q: Float[Array, "B H L d_k"] = rearrange(
            self.Wq(query), "B L (H d_k) -> B H L d_k"
        )
        K: Float[Array, "B H L d_k"] = rearrange(
            self.Wk(key), "B L (H d_k) -> B H L d_k"
        )
        V: Float[Array, "B H L d_k"] = rearrange(
            self.Wv(value), "B L (H d_k) -> B H L d_k"
        )

        scores: Float[Array, "B H L L"] = einsum(
            Q, K, "B H L d_k, B H L d_k -> B H L L"
        ) / jnp.sqrt(self.head_dim)
        softmax: Float[Array, "B H L L"] = jax.nn.softmax(scores, axis=-1, where=mask)
        attention: Float[Array, "B H L d_k"] = einsum(
            softmax, V, "B H L L, B H L d_k -> B H L d_k"
        )
        multi_attention: Float[Array, "B L D"] = rearrange(
            attention, "B H L d_k -> B L (H d_k)"
        )
        return self.Wo(multi_attention)


class Embedding(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        seq_length: int,
        dropout_prob: float,
        rngs: nnx.Rngs,
    ):
        self.Embedding: nnx.Embed = nnx.Embed(vocab_size, hidden_size, rngs=rngs)

        den = jnp.exp(-jnp.arange(0, hidden_size, 2) * jnp.log(10000.0) / hidden_size)
        pos: Int[Array, "L 1"] = jnp.arange(0, seq_length)[:, jnp.newaxis]
        pe: Float[Array, "L D"] = jnp.zeros((seq_length, hidden_size))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * den))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * den))
        self.pos_embedding = nnx.Variable(pe)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, tokens: Int[Array, "B L"]) -> Float[Array, "B L D"]:
        curr_seqlength: int = tokens.shape[1]
        token_embedding: Float[Array, "B L D"] = self.Embedding(tokens)
        return self.dropout(
            token_embedding + self.pos_embedding.value[:curr_seqlength, :]
        )


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
        x: Float[Array, "B L D"],
        encoder_output: Optional[Float[Array, "B L D"]],
        self_mask: Bool[Array, "B 1 L L"],
        cross_mask: Optional[Bool[Array, "B 1 L L"]],
    ) -> Float[Array, "B L D"]:
        norm: Float[Array, "B L D"] = self.norm1(x)
        x += self.self_attention(norm, norm, norm, self_mask)

        if encoder_output is not None and cross_mask is not None:
            norm: Float[Array, "B L D"] = self.norm2(x)
            x += self.cross_attention(norm, encoder_output, encoder_output, cross_mask)

        norm: Float[Array, "B L D"] = self.norm3(x)
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
        self.embedding = Embedding(
            hidden_size, vocab_size, seq_length, dropout_prob, rngs
        )
        self.decoder = nnx.List(
            [
                Transformer(hidden_size, interm_size, num_heads, dropout_prob, rngs)
                for _ in range(layers)
            ]
        )
        self.out = nnx.Linear(hidden_size, vocab_size, rngs=rngs)

    def __call__(
        self,
        tokens: Int[Array, "B L"],
        encoder_output: Optional[Float[Array, "B L D"]],
        self_mask: Bool[Array, "B 1 L L"],
        cross_mask: Optional[Bool[Array, "B 1 L L"]],
    ):
        hidden_output: Float[Array, "B L D"] = self.embedding(tokens)
        for layer in self.decoder:
            hidden_output: Float[Array, "B L D"] = layer(
                hidden_output, encoder_output, self_mask, cross_mask
            )
        return self.out(hidden_output)
