from jaxtyping import install_import_hook
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Array, Bool, Int
import jax
import optax
import argparse
import datasets as dats
from torch.utils.data import DataLoader
import orbax.checkpoint as ocp
import os

from model import DecoderModel

hook = install_import_hook("TransformerDecoder", "beartype.beartype")


@nnx.jit
def train_batch(batch, model: DecoderModel, optimizer, source_pad_idx, target_pad_idx):
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

        source_padding_mask: Bool[Array, "B 1 1 L1"] = (source != source_pad_idx)[
            :, jnp.newaxis, jnp.newaxis, :
        ]
        target_padding_mask: Bool[Array, "B 1 1 L2"] = (
            target_shifted != target_pad_idx
        )[:, jnp.newaxis, jnp.newaxis, :]
        padding_mask: Bool[Array, "B 1 1 L"] = jnp.concatenate(
            [source_padding_mask, target_padding_mask], axis=-1
        )

        attention_mask: Bool[Array, "B 1 L L"] = causal_mask & padding_mask

        model_logits: Float[Array, "B L V"] = model(
            full_input, None, attention_mask, None
        )
        predicted_logits: Float[Array, "B L2 V"] = model_logits[:, L1:, :]
        target_loss: Float[Array, "B L2"] = (
            optax.softmax_cross_entropy_with_integer_labels(predicted_logits, target)
        )
        loss_mask: Bool[Array, "B L2"] = target != target_pad_idx

        return jnp.sum(target_loss * loss_mask) / jnp.sum(loss_mask), predicted_logits

    (loss, logits), grads = nnx.value_and_grad(calculate_loss, has_aux=True)(model)
    optimizer.update(model, grads)

    correct = jnp.argmax(logits, axis=-1) == target
    accuracy = jnp.sum(correct * (target != target_pad_idx)) / jnp.sum(
        target != target_pad_idx
    )
    return loss, accuracy


@nnx.jit
def val_batch(batch, model: DecoderModel, source_pad_idx, target_pad_idx):
    source: Int[Array, "B L1"] = batch["xq_context_padded"]
    target_shifted: Int[Array, "B L2"] = batch["yq_sos_padded"]
    target: Int[Array, "B L2"] = batch["yq_padded"]
    full_input: Int[Array, "B L"] = jnp.concatenate([source, target_shifted], axis=1)
    B, L = full_input.shape
    _, L1 = source.shape

    causal_mask: Bool[Array, "B 1 L L"] = jnp.tril(jnp.ones((B, 1, L, L), dtype=bool))
    source_padding_mask: Bool[Array, "B 1 1 L1"] = (source != source_pad_idx)[
        :, jnp.newaxis, jnp.newaxis, :
    ]
    target_padding_mask: Bool[Array, "B 1 1 L2"] = (target_shifted != target_pad_idx)[
        :, jnp.newaxis, jnp.newaxis, :
    ]
    padding_mask: Bool[Array, "B 1 1 L"] = jnp.concatenate(
        [source_padding_mask, target_padding_mask], axis=-1
    )
    attention_mask: Bool[Array, "B 1 L L"] = causal_mask & padding_mask

    model_logits: Float[Array, "B L V"] = model(full_input, None, attention_mask, None)
    predicted_logits: Float[Array, "B L2 V"] = model_logits[:, L1:, :]
    target_loss: Float[Array, "B L2"] = optax.softmax_cross_entropy_with_integer_labels(
        predicted_logits, target
    )
    loss_mask: Bool[Array, "B L2"] = target != target_pad_idx

    correct = jnp.argmax(predicted_logits, axis=-1) == target
    accuracy = jnp.sum(correct * (target != target_pad_idx)) / jnp.sum(
        target != target_pad_idx
    )

    return jnp.sum(target_loss * loss_mask) / jnp.sum(loss_mask), accuracy


def validate(model, val_loader, source_pad_idx, target_pad_idx):
    model.eval()
    avg_loss, avg_acc = 0.0, 0.0
    for batch in val_loader:
        jax_batch = jax.tree_util.tree_map(
            lambda x: jnp.array(x.numpy()) if hasattr(x, "numpy") else x, batch
        )
        input_keys = ["xq_context_padded", "yq_sos_padded", "yq_padded"]
        clean_batch = {k: jax_batch[k] for k in input_keys}
        loss, acc = val_batch(clean_batch, model, source_pad_idx, target_pad_idx)
        avg_loss += loss
        avg_acc += acc
    model.train()
    return avg_loss / len(val_loader), avg_acc / len(val_loader)


def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    num_epochs,
    source_pad_idx,
    target_pad_idx,
    log_every,
    val_every,
    mngr,
    checkpoint_dir,
):
    best_val_loss = float("inf")
    val_loss = float("inf")
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()

        for batch in train_loader:
            jax_batch = jax.tree_util.tree_map(
                lambda x: jnp.array(x.numpy()) if hasattr(x, "numpy") else x, batch
            )
            input_keys = ["xq_context_padded", "yq_sos_padded", "yq_padded"]
            clean_batch = {k: jax_batch[k] for k in input_keys}
            loss, acc = train_batch(
                clean_batch, model, optimizer, source_pad_idx, target_pad_idx
            )
            if global_step % log_every == 0:
                print(
                    f"Step {global_step} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}"
                )
            if global_step % val_every == 0:
                val_loss, val_acc = validate(
                    model, val_loader, source_pad_idx, target_pad_idx
                )
                print(
                    f"VALIDATION | Loss: {val_loss.item():.4f} | Acc: {val_acc.item():.4f}"
                )

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(mngr, checkpoint_dir, model, optimizer, global_step)
            global_step += 1

    print("Training complete.")


def save_checkpoint(mngr, checkpoint_dir, model, optimizer, step):
    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
    }

    mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state)))

    mngr.wait_until_finished()
    print(f" > Checkpoint saved at step {step}")


def load_checkpoint(mngr, model, optimizer):
    latest_step = mngr.latest_step()
    if latest_step is None:
        print("No checkpoint found. Starting from scratch.")
        return
    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": latest_step,
    }
    restored = mngr.restore(
        latest_step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )

    nnx.update(model, restored["state"]["model"])
    nnx.update(optimizer, restored["state"]["optimizer"])

    print(f" > Restored checkpoint from step {latest_step}")
    return restored["state"]["step"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_model",
        type=str,
        default="out_models",
        help="Directory for saving model files",
    )
    parser.add_argument(
        "--episode_type",
        type=str,
        default="retrieve",
        help="What type of episodes do we want? See datasets.py for options",
    )
    parser.add_argument(
        "--batch_size", type=int, default=25, help="number of episodes per batch"
    )
    parser.add_argument(
        "--nepochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="number of attention heads"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay"
    )
    parser.add_argument(
        "--nlayers_decoder", type=int, default=3, help="number of layers for decoder"
    )
    parser.add_argument("--emb_size", type=int, default=128, help="size of embedding")
    parser.add_argument(
        "--ff_mult",
        type=int,
        default=4,
        help="multiplier for size of the fully-connected layer in transformer",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout applied to embeddings and transformer",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume training from a previous checkpoint",
    )
    parser.add_argument(
        "--save_best",
        default=False,
        action="store_true",
        help='Save the "best model" according to validation loss.',
    )

    args = parser.parse_args()
    hidden_size = args.emb_size
    interm_size = args.ff_mult * hidden_size
    dropout_prob = args.dropout
    nlayers = args.nlayers_decoder
    learning_rate = args.lr
    num_epochs = args.nepochs
    batch_size = args.batch_size
    episode_type = args.episode_type
    num_heads = args.num_heads
    weight_decay = args.weight_decay
    save_best = args.save_best
    log_every = 10
    val_every = 100
    seq_length = 5000
    b1 = 0.9
    b2 = 0.95

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
    source_pad_idx = langs["input"].PAD_idx
    target_pad_idx = langs["output"].PAD_idx
    source_vocab_size = langs["input"].n_symbols
    target_vocab_size = langs["output"].n_symbols

    checkpoint_dir = os.path.abspath(args.dir_model)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(checkpoint_dir, item_names=("state",), options=options)

    rngs = nnx.Rngs(42)
    model = DecoderModel(
        hidden_size,
        interm_size,
        num_heads,
        source_vocab_size,
        target_vocab_size,
        seq_length,
        nlayers,
        dropout_prob,
        rngs,
    )
    tx = optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    train(
        model,
        optimizer,
        train_loader,
        val_loader,
        num_epochs,
        source_pad_idx,
        target_pad_idx,
        log_every,
        val_every,
        mngr,
        checkpoint_dir,
    )
