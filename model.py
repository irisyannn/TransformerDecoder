from flax import nnx
from jaxtyping import Float, Array, Bool, Int
import jax.numpy as jnp
import jax
from einops import rearrange, einsum

class Attention(nnx.Module):
    def __init__(self, num_heads: int, hidden_size: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        head_dim = self.hidden_size // self.num_heads
        self.Wq = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wk = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wv = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.Wo = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, query: Float[Array, "B Lq D"], key: Float[Array, "B Lk D"], value: Float[Array, "B Lk D"], mask: Bool[Array, "B 1 Lq Lk"]) -> Float[Array, "B Lq D"]:
        B: int = query.shape[0]
        Lq: int = query.shape[1]
        Lk: int = key.shape[1]
        
        Q: Float[Array, "B H Lq d_k"] = rearrange(self.Wq(query), 'B Lq (H d_k) -> B H Lq d_k')
        K: Float[Array, "B H Lk d_k"] = rearrange(self.Wk(key), 'B Lk (H d_k) -> B H Lk d_k')
        V: Float[Array, "B H Lk d_k"] = rearrange(self.Wv(value), 'B Lk (H d_k) -> B H Lk d_k')

        scores: Float[Array, 'B H Lq Lk'] = einsum(Q, K, "B H Lq d_k, B H Lk d_k -> B H Lq Lk") / jnp.sqrt(self.head_dim)
        softmax: Float[Array, 'B H Lq Lk'] = jax.nn.softmax(scores, axis=-1, where=mask)
        attention: Float[Array, 'B H Lq d_k'] = einsum(softmax, V, 'B H Lq Lk, B H Lk d_k -> B H Lq d_k')
        multi_attention: Float[Array, 'B Lq D'] = rearrange(attention, 'B H Lq d_k -> B Lq (H d_k)')
        return self.Wo(multi_attention)

class Embedding(nnx.Module):
    def __init__(self, hidden_size: int, vocab_size: int, seq_length: int, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.Embedding: Float[Array, 'V D'] = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
        self.Positional: Float[Array, 'L D'] = nnx.Embed(seq_length, hidden_size, rngs=rngs)

    def __call__(self, tokens: Int[Array, 'B L']) -> Float[Array, 'B L D']:
        positions: Int[Array, 'B L'] = jnp.arange(self.seq_length)[None, :]
        return self.Embedding(tokens) + self.Positional(positions)

class DecoderModel(nnx.Module):
    def __init__(self):
        