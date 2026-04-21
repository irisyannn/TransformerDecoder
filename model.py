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
        self,
        hidden_size: int,
        vocab_size: int,
        seq_length: int,
        dropout_prob: float,
        rngs: nnx.Rngs,
    ):
        self.Embedding: nnx.Embed = nnx.Embed(vocab_size, hidden_size, rngs=rngs)

        den = jnp.exp(-jnp.arange(0, hidden_size, 2) * jnp.log(10000.0) / hidden_size)
        pos: Int[Array, "Lq 1"] = jnp.arange(0, seq_length)[:, jnp.newaxis]
        pe: Float[Array, "Lq D"] = jnp.zeros((seq_length, hidden_size))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * den))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * den))
        self.pos_embedding = nnx.Variable(pe)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, tokens: Int[Array, "B Lq"]) -> Float[Array, "B Lq D"]:
        curr_seqlength = tokens.shape[1]
        token_embedding = self.Embedding(tokens)
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
        x: Float[Array, "B Lq D"],
        encoder_output: Float[Array, "B Lk D"],
        self_mask: Bool[Array, "B 1 Lq Lk"],
        cross_mask: Bool[Array, "B 1 Lq Lk"],
    ) -> Float[Array, "B Lq D"]:
        norm: Float[Array, "B L D"] = self.norm1(x)
        x += self.self_attention(norm, norm, norm, self_mask)

        norm: Float[Array, "B Lq D"] = self.norm2(x)
        x += self.cross_attention(norm, encoder_output, encoder_output, cross_mask)

        norm: Float[Array, "B Lq D"] = self.norm3(x)
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
        tokens: Int[Array, "B Lq V"],
        encoder_output: Float[Array, "B Lk D"],
        self_mask: Bool[Array, "B 1 Lq Lk"],
        cross_mask: Bool[Array, "B 1 Lq Lk"],
    ):
        hidden_output: Float[Array, "B Lq D"] = self.embedding(tokens)
        for layer in self.decoder:
            hidden_output: Float[Array, "B Lq D"] = layer(
                hidden_output, encoder_output, self_mask, cross_mask
            )
        return self.out(hidden_output)
