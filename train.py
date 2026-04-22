from jaxtyping import install_import_hook
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Array, Bool, Int
import jax
import optax
import argparse
import datasets as dats
from torch.utils.data import DataLoader

from model import DecoderModel

hook = install_import_hook("TransformerDecoder", "beartype.beartype")


@nnx.jit
def train_batch(batch, model: DecoderModel, loss_fn, optimizer, pad_idx):
    source: Int[Array, "B L1"] = batch["xq_context_padded"]
    target_shifted: Int[Array, "B L2"] = batch["yq_sos_padded"]
    target: Int[Array, "B L2"] = batch["yq_padded"]
    full_input: Int[Array, "B L"] = jnp.concatenate([source, target_shifted], axis=1)
    B, L = full_input.shape
    _, L1 = source.shape

    def calculate_loss(model):
        causal_mask: Bool[Array, "B 1 L L"] = jnp.tril(
            jnp.ones((B, 1, L, L), dtype=bool)
        )
        padding_mask: Bool[Array, "B 1 1 L"] = (full_input != pad_idx)[
            :, jnp.newaxis, jnp.newaxis, :
        ]
        attention_mask: Bool[Array, "B 1 L L"] = causal_mask & padding_mask

        model_logits: Float[Array, "B L V"] = model(
            full_input, None, attention_mask, None
        )
        predicted_logits: Int[Array, "B L2 V"] = model_logits[:, L1 - 1 : -1, :]
        target_loss: Float[Array, "B L2"] = loss_fn(predicted_logits, target)
        loss_mask: Bool[Array, "B L2"] = target != pad_idx

        return (target_loss * loss_mask) / jnp.sum(loss_mask), predicted_logits

    (loss, logits), grads = nnx.value_and_grad(calculate_loss, has_aux=True)(model)
    optimizer.update(model, grads)

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == target)
    return loss, accuracy


def val_batch(batch, model: DecoderModel, loss_fn, pad_idx):
    source: Int[Array, "B L1"] = batch["xq_context_padded"]
    target_shifted: Int[Array, "B L2"] = batch["yq_sos_padded"]
    target: Int[Array, "B L2"] = batch["yq_padded"]
    full_input: Int[Array, "B L"] = jnp.concatenate([source, target_shifted], axis=1)
    B, L = full_input.shape
    _, L1 = source.shape

    causal_mask: Bool[Array, "B 1 L L"] = jnp.tril(jnp.ones((B, 1, L, L), dtype=bool))
    padding_mask: Bool[Array, "B 1 1 L"] = (full_input != pad_idx)[
        :, jnp.newaxis, jnp.newaxis, :
    ]
    attention_mask: Bool[Array, "B 1 L L"] = causal_mask & padding_mask

    model_logits: Float[Array, "B L V"] = model(full_input, None, attention_mask, None)
    predicted_logits: Int[Array, "B L2"] = model_logits[:, L1 - 1 : -1, :]
    target_loss: Float[Array, "B L2"] = loss_fn(predicted_logits, target)
    loss_mask: Bool[Array, "B L2"] = target != pad_idx

    accuracy = jnp.mean(jnp.argmax(predicted_logits, axis=-1) == target)

    return (target_loss * loss_mask) / jnp.sum(loss_mask), accuracy


def validate(model, val_loader, loss_fn, pad_idx):
    avg_loss, avg_acc = 0.0, 0.0
    for batch in val_loader:
        jax_batch = jax.tree_util.tree_map(
            lambda x: jnp.array(x.numpy()) if hasattr(x, "numpy") else x, batch
        )
        loss, acc = val_batch(jax_batch, model, loss_fn, pad_idx)
        avg_loss += loss
        avg_acc += acc
    return avg_loss / len(val_loader), avg_acc / len(val_loader)


def train(
    model, train_loader, val_loader, loss_fn, num_epochs, pad_idx, log_every, val_every
):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()

        for step, batch in enumerate(train_loader):
            jax_batch = jax.tree_util.tree_map(
                lambda x: jnp.array(x.numpy()) if hasattr(x, "numpy") else x, batch
            )
            loss, acc = train_batch(jax_batch, model, loss_fn, optimizer, pad_idx)
            if step % log_every == 0:
                print(f"Step {step} | Loss: {loss:.4f} | Acc: {acc:.4f}")
            if step % val_every == 0:
                val_loss, val_acc = validate(model, val_loader, loss_fn, pad_idx)
                print(f"VALIDATION | Loss: {loss:.4f} | Acc: {acc:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    hidden_size = args.hidden_size
    interm_size = args.interm_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    seq_length = args.seq_length
    dropout_prob = args.dropout_prob
    nlayers = args.nlayers
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    log_every = args.log_every
    val_every = args.val_every
    episode_type = args.episode_type

    D_train, D_val = dats.get_dataset(episode_type)
    train_loader = DataLoader(
        D_train,
        batch_size=batch_size,
        collate_fn=lambda x: dats.make_biml_batch(x, D_train.langs),
        shuffle=True,
    )
    val_loader = DataLoader(
        D_val,
        batch_size=batch_size,
        collate_fn=lambda x: dats.make_biml_batch(x, D_val.langs),
        shuffle=False,
    )
    langs = D_train.langs
    pad_idx = langs["output"].PAD_idx

    rngs = nnx.Rngs(42)
    model = DecoderModel(
        hidden_size,
        interm_size,
        num_heads,
        vocab_size,
        seq_length,
        nlayers,
        dropout_prob,
        rngs,
    )
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    loss_fn = optax.softmax_cross_entropy_with_integer_labels

    train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        num_epochs,
        pad_idx,
        log_every,
        val_every,
    )
