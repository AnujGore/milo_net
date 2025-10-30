import jax
import numpy as np
import jax.numpy as jnp
import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import optax
import time

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # Moves one directory up
sys.path.append(project_root)

from src.model import Milo, CNN
from src.train import create_state_MLP
from src.utils import preprocess


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits = jnp.squeeze(logits), labels = batch[1]).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    loss = optax.softmax_cross_entropy_with_integer_labels(jnp.squeeze(logits), labels = batch[1]).mean()
    accuracy = jnp.mean(jnp.argmax(jnp.squeeze(logits), -1) == batch[1])

    return state, loss, accuracy
        
@jax.jit
def eval_step(state, label):
    preds = state.apply_fn({"params": state.params}, label)
    return preds

def evaluate_model(state, val_ds):
    val_losses = []
    val_accuracies = []
    
    for batch in val_ds.as_numpy_iterator():
        inputs, labels = batch
        preds = eval_step(state, inputs)
        
        loss = optax.softmax_cross_entropy_with_integer_labels(jnp.squeeze(preds), labels = batch[1]).mean()
        accuracy = np.mean(np.argmax(jnp.squeeze(preds), axis=-1) == labels)
        
        val_losses.append(loss)
        val_accuracies.append(accuracy)

    return val_losses, val_accuracies


rng = jax.random.PRNGKey(0)
device = jax.devices("cpu")[0] #Currently CPU

#Load and normalize the dataset
train_ds: tf.data.Dataset = tfds.load('cifar10', split='train')
train_ds_cnn = train_ds.map(preprocess).cache().shuffle(1024).batch(32, drop_remainder=True).take(2500).prefetch(1)
train_ds_milo = train_ds.map(preprocess).cache().shuffle(1024).batch(32, drop_remainder=True).take(2500).prefetch(1)

val_ds: tf.data.Dataset = tfds.load('cifar10', split='test')
val_ds_cnn = val_ds.map(preprocess).cache().batch(32).prefetch(1)
val_ds_milo = val_ds.map(preprocess).cache().batch(32).prefetch(1)

# =============================
# Hyperparameters
# =============================
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 200
PATIENCE = 50

# =============================
# Model Instantiation
# =============================


milo = Milo(
    input_dim=(32, 32), 
    hidden_layer_dim=[(64, 64), (128, 128), (256, 256), (512, 512), (256, 256), (128, 128), (64, 64), (36, 36), (18, 16), (10, 1)],
    channel_output_dim = (10, 1),
    output_dim=10,
    num_channels=3
)


# CNN baseline
cnn_model = CNN()

# =============================
# Create states
# =============================
milo_state = create_state_MLP(rng, milo, LR, data_size=(BATCH_SIZE, 32, 32, 3), device=device)
cnn_state = create_state_MLP(rng, cnn_model, LR, data_size=(BATCH_SIZE, 32, 32, 3), device=device)

del rng

# =============================
# Tracking
# =============================
patience_counters = {"milo": 0, "cnn": 0}
best_val_losses = {"milo": np.inf, "cnn": np.inf}
best_states = {"milo": None, "cnn": None}

metrics_history = {k: [] for k in [
    'milo_train_loss', 'milo_rgb_train_loss_std',
    'milo_train_accuracy', 'milo_rgb_train_accuracy_std',
    'milo_val_loss', 'milo_rgb_val_loss_std',
    'milo_val_accuracy', 'milo_rgb_val_accuracy_std',
    'milo_time', 'milo_rgb_time_std',

    'cnn_train_loss', 'cnn_train_loss_std',
    'cnn_train_accuracy', 'cnn_train_accuracy_std',
    'cnn_val_loss', 'cnn_val_loss_std',
    'cnn_val_accuracy', 'cnn_val_accuracy_std',
    'cnn_time', 'cnn_time_std'
]}

# =============================
# Training Loop
# =============================
with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn()
) as progress:

    # Progress bar tasks
    tasks = {
        "milo": progress.add_task("Training Milo...", total=NUM_EPOCHS),
        "cnn": progress.add_task("Training CNN...", total=NUM_EPOCHS)
    }

    for epoch in range(NUM_EPOCHS):
        # --- Epoch metrics ---
        epoch_metrics = {
            "milo": {"loss": [], "acc": [], "time": []},
            "cnn": {"loss": [], "acc": [], "time": []}
        }

        # --- Training pass ---
        for (milo_batch, cnn_batch) in zip(train_ds_milo.as_numpy_iterator(),
                                           train_ds_cnn.as_numpy_iterator()):
            milo_x, milo_y = milo_batch
            cnn_x, cnn_y = cnn_batch

            # MiloRGB model training
            start = time.time()
            milo_state, loss, acc = train_step(milo_state, (milo_x, milo_y))
            epoch_metrics["milo"]["time"].append(time.time() - start)
            epoch_metrics["milo"]["loss"].append(loss)
            epoch_metrics["milo"]["acc"].append(acc)

            # CNN model training
            start = time.time()
            cnn_state, loss, acc = train_step(cnn_state, (cnn_x, cnn_y))
            epoch_metrics["cnn"]["time"].append(time.time() - start)
            epoch_metrics["cnn"]["loss"].append(loss)
            epoch_metrics["cnn"]["acc"].append(acc)

        # --- Update progress ---
        for model_key in ["milo", "cnn"]:
            progress.update(
                tasks[model_key],
                advance=1,
                description=f"Training {model_key}... "
                            f"Loss: {np.mean(epoch_metrics[model_key]['loss']):.4f} ± {np.std(epoch_metrics[model_key]['loss']):.4f} "
                            f"Acc: {np.mean(epoch_metrics[model_key]['acc']):.4f} ± {np.std(epoch_metrics[model_key]['acc']):.4f}"
            )

        # --- Validation ---
        for model_key, (val_ds_used, state) in [
            ("milo", (val_ds_milo, milo_state)),
            ("cnn", (val_ds_cnn, cnn_state))
        ]:
            val_loss, val_acc = evaluate_model(state, val_ds_used)

            if np.mean(val_loss) < best_val_losses[model_key]:
                best_val_losses[model_key] = np.mean(val_loss)
                best_states[model_key] = state
                patience_counters[model_key] = 0
            else:
                patience_counters[model_key] += 1

            # Save metrics
            metrics_history[f"{model_key}_train_loss"].append(np.mean(epoch_metrics[model_key]["loss"]))
            metrics_history[f"{model_key}_train_loss_std"].append(np.std(epoch_metrics[model_key]["loss"]))
            metrics_history[f"{model_key}_train_accuracy"].append(np.mean(epoch_metrics[model_key]["acc"]))
            metrics_history[f"{model_key}_train_accuracy_std"].append(np.std(epoch_metrics[model_key]["acc"]))
            metrics_history[f"{model_key}_val_loss"].append(np.mean(val_loss))
            metrics_history[f"{model_key}_val_loss_std"].append(np.std(val_loss))
            metrics_history[f"{model_key}_val_accuracy"].append(np.mean(val_acc))
            metrics_history[f"{model_key}_val_accuracy_std"].append(np.std(val_acc))
            metrics_history[f"{model_key}_time"].append(np.mean(epoch_metrics[model_key]["time"]))
            metrics_history[f"{model_key}_time_std"].append(np.std(epoch_metrics[model_key]["time"]))

        # --- Early stopping ---
        if all(patience_counters[m] >= PATIENCE for m in patience_counters):
            print("Early stopping triggered for all models.")
            break

# =============================
# Save models
# =============================
os.makedirs("models", exist_ok=True)
from flax.serialization import to_bytes
import json

for model_key in best_states:
    with open(f"models/{model_key}_cifar_best_state.msgpack", "wb") as f:
        f.write(to_bytes(best_states[model_key]))

# Save metrics
def convert_metrics_to_serializable(metrics):
    return {k: list(map(float, v)) for k, v in metrics.items()}

with open("models/cifar_metrics_history.json", "w") as f:
    json.dump(convert_metrics_to_serializable(metrics_history), f)