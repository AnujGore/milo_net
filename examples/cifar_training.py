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

from src.model import MiloMLP, CNN, MiloWCon
from src.train import create_state_MLP
from src.utils import preprocess


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = batch[1]).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels = batch[1]).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch[1])

    return state, loss, accuracy
        
@jax.jit
def eval_step(state, label):
    preds = state.apply_fn({"params": state.params}, label)
    return preds

rng = jax.random.PRNGKey(0)
device = jax.devices("cpu")[0] #Currently CPU

#Load and normalize the dataset
train_ds: tf.data.Dataset = tfds.load('cifar10', split='train')
train_ds = train_ds.map(preprocess).cache().shuffle(1024).batch(32, drop_remainder=True).take(2500).prefetch(1)

#Hyperparameters
BATCH_SIZE = 32
lr = 1e-3
NUM_EPOCHS = 100

milo_con_multichannel = MiloWCon(
    input_dim=(8, 8), 
    hidden_layer_dim=[(16, 16), (10, 1)],
    output_dim=(10, 1),
    num_channels=3
)

milo_multichannel = MiloMLP(
    input_dim=(32, 32), 
    hidden_layer_dim=[(64, 64), (32, 32), (24, 18), (16, 9), (10, 1)],
    output_dim=(10, 1),
    num_channels=3
)

cnn_model = CNN()

milo_multichannel_state = create_state_MLP(rng, milo_multichannel, lr, data_size=(BATCH_SIZE, 32, 32, 3), device=device)
milo_con_multichannel_state = create_state_MLP(rng, milo_con_multichannel, lr, data_size=(BATCH_SIZE, 32, 32, 3), device=device)
cnn_state = create_state_MLP(rng, cnn_model, lr, data_size=(BATCH_SIZE, 32, 32, 3), device=device)

del rng

milo_best_loss = np.inf; cnn_best_loss = np.inf; milo_con_best_loss = np.inf
milo_best_state = None; cnn_best_state = None; milo_con_best_state = None

metrics_history = {
    'milo_loss': [], 'milo_loss_std': [],
    'milo_accuracy': [], 'milo_accuracy_std': [],
    'milo_time': [], 'milo_time_std': [],
    'milo_con_loss': [], 'milo_con_loss_std': [],
    'milo_con_accuracy': [], 'milo_con_accuracy_std': [],
    'milo_con_time': [], 'milo_con_time_std': [],
    'cnn_loss': [], 'cnn_loss_std': [],
    'cnn_accuracy': [], 'cnn_accuracy_std': [],
    'cnn_time': [], 'cnn_time_std': [],
}



with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=True  # hides progress bars after completion
) as progress:

    milo_task = progress.add_task("Training Milo Model...", total=NUM_EPOCHS)
    milo_con_task = progress.add_task("Training Milo (with Convolutions) Model...", total=NUM_EPOCHS)
    cnn_task = progress.add_task("Training CNN Model...", total=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        milo_loss, milo_accuracy, milo_time = [], [], []
        milo_con_loss, milo_con_accuracy, milo_con_time = [], [], []
        cnn_loss, cnn_accuracy, cnn_time = [], [], []
        
        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            labels = np.expand_dims(batch[1], axis=(1))

            # Milo training step
            start_time = time.time()
            milo_multichannel_state, this_loss, accuracy = train_step(milo_multichannel_state, batch)
            milo_time.append(time.time() - start_time)
            milo_loss.append(this_loss)
            milo_accuracy.append(accuracy)

        progress.update(milo_task, advance=1, description=f"Training Milo Model... Loss: {np.mean(milo_loss):.4f} ± {np.std(milo_loss):.4f}")

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            labels = np.expand_dims(batch[1], axis=(1))

            # Milo training step
            start_time = time.time()
            milo_con_multichannel_state, this_loss, accuracy = train_step(milo_con_multichannel_state, batch)
            milo_con_time.append(time.time() - start_time)
            milo_con_loss.append(this_loss)
            milo_con_accuracy.append(accuracy)

        progress.update(milo_con_task, advance=1, description=f"Training Milo (with Convolutions) Model... Loss: {np.mean(milo_con_loss):.4f} ± {np.std(milo_con_loss):.4f}")

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            labels = np.expand_dims(batch[1], axis=(1))

            # CNN training step
            start_time = time.time()
            cnn_state, this_loss, accuracy = train_step(cnn_state, batch)
            cnn_time.append(time.time() - start_time)
            cnn_loss.append(this_loss)
            cnn_accuracy.append(accuracy)

        progress.update(cnn_task, advance=1, description=f"Training CNN Model... Loss: {np.mean(cnn_loss):.4f} ± {np.std(cnn_loss):.4f}")


        metrics_history['milo_loss'].append(np.mean(milo_loss))
        metrics_history['milo_loss_std'].append(np.std(milo_loss))
        metrics_history['milo_accuracy'].append(np.mean(milo_accuracy))
        metrics_history['milo_accuracy_std'].append(np.std(milo_accuracy))
        metrics_history['milo_time'].append(np.mean(milo_time))
        metrics_history['milo_time_std'].append(np.std(milo_time))

        metrics_history['milo_con_loss'].append(np.mean(milo_con_loss))
        metrics_history['milo_con_loss_std'].append(np.std(milo_con_loss))
        metrics_history['milo_con_accuracy'].append(np.mean(milo_con_accuracy))
        metrics_history['milo_con_accuracy_std'].append(np.std(milo_con_accuracy))
        metrics_history['milo_con_time'].append(np.mean(milo_con_time))
        metrics_history['milo_con_time_std'].append(np.std(milo_con_time))

        metrics_history['cnn_loss'].append(np.mean(cnn_loss))
        metrics_history['cnn_loss_std'].append(np.std(cnn_loss))
        metrics_history['cnn_accuracy'].append(np.mean(cnn_accuracy))
        metrics_history['cnn_accuracy_std'].append(np.std(cnn_accuracy))
        metrics_history['cnn_time'].append(np.mean(cnn_time))
        metrics_history['cnn_time_std'].append(np.std(cnn_time))

        if np.mean(milo_loss) < milo_best_loss:
            milo_best_loss = np.mean(milo_loss)
            milo_best_state = milo_multichannel_state
        
        if np.mean(milo_con_loss) < milo_con_best_loss:
            milo_con_best_loss = np.mean(milo_con_loss)
            milo_con_best_state = milo_con_multichannel_state

        if np.mean(cnn_loss) < cnn_best_loss:
            cnn_best_loss = np.mean(cnn_loss)
            cnn_best_state = cnn_state

from flax.serialization import to_bytes
import json

os.makedirs("models", exist_ok=True)

with open("models/milo_cifar_best_state.msgpack", "wb") as f:
    f.write(to_bytes(milo_best_state))

with open("models/cnn_cifar_best_state.msgpack", "wb") as f:
    f.write(to_bytes(cnn_best_state))

with open("models/milo_con_cifar_best_state.msgpack", "wb") as f:
    f.write(to_bytes(milo_con_best_state))

# --- Save metrics to JSON ---
def convert_metrics_to_serializable(metrics):
    return {k: list(map(float, v)) for k, v in metrics.items()}

with open("models/cifar_metrics_history.json", "w") as f:
    json.dump(convert_metrics_to_serializable(metrics_history), f)