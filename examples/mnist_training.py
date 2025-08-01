import jax
import numpy as np
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import time
tf.random.set_seed(0) 
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # Moves one directory up
sys.path.append(project_root)

from src.model import MiloMLP, CNN
from src.train import create_state_MLP

@jax.jit
def train_step_milo(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        logits = jnp.squeeze(logits, -1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits.squeeze(), labels = batch['label']).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    return state, metrics

@jax.jit
def train_step_cnn(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = batch['label']).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    return state, metrics

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch['image'])
    return compute_metrics(logits, batch['label'])


@jax.jit
def pred_label(state, batch):
    preds = state.apply_fn({"params": state.params}, batch["image"])
    return preds.argmax(axis = 1)

def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

rng = jax.random.PRNGKey(0)
device = jax.devices("cpu")[0] #Currently CPU

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.squeeze(tf.cast(sample['image'], tf.float32) / 255, -1),
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.squeeze(tf.cast(sample['image'], tf.float32) / 255, -1),
    'label': sample['label'],
  }
)  # Normalize the test set.

lr = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 32

train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).take(2500).cache().prefetch(1)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

milo = MiloMLP(
    input_dim=(28, 28), 
    hidden_layer_dim=[(64, 64), (36, 36), (18, 16), (10, 1)],
    output_dim=(10, 1),
    num_channels=1 
)

simpleCNN = CNN()

milo_state = create_state_MLP(rng, milo, lr, data_size=(32, 28, 28), device=device)
cnn_state = create_state_MLP(rng, simpleCNN, lr, data_size=(32, 28, 28, 1), device=device)

del rng

metrics_history = {
    'milo_loss': [], 'milo_loss_std': [],
    'milo_accuracy': [], 'milo_accuracy_std': [],
    'milo_time': [], 'milo_time_std': [],
    'cnn_loss': [], 'cnn_loss_std': [],
    'cnn_accuracy': [], 'cnn_accuracy_std': [],
    'cnn_time': [], 'cnn_time_std': [],
}

milo_best_loss = np.inf
cnn_best_loss = np.inf
milo_best_state = None
cnn_best_state = None

with Progress( TextColumn("[bold blue]{task.description}"),BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as progress:
    
    milo_task = progress.add_task("Training Milo Model...", total=NUM_EPOCHS)
    cnn_task = progress.add_task("Training CNN Model...", total=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):

        # Per-epoch tracking
        milo_loss = []; milo_accuracy = []; milo_time = []
        cnn_loss = []; cnn_accuracy = []; cnn_time = []

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            # Milo training step
            start_time = time.time()
            milo_state, milo_metrics = train_step_milo(milo_state, batch)
            milo_time.append(time.time() - start_time)
            milo_accuracy.append(milo_metrics["accuracy"])
            milo_loss.append(milo_metrics["loss"])

        progress.update(milo_task, advance=1, description=f"Training Milo Model... Loss: {np.mean(milo_loss):.4f} ± {np.std(milo_loss):.4f}")

        for step, this_batch in enumerate(train_ds.as_numpy_iterator()):
            # CNN training (with extra channel dim added)
            this_batch = {
                'image': this_batch["image"][..., None],
                'label': this_batch["label"]
            }
            start_time = time.time()  # Start timing
            cnn_state, cnn_metrics = train_step_cnn(cnn_state, this_batch)
            cnn_time.append(time.time() - start_time)            
            cnn_accuracy.append(cnn_metrics["accuracy"])
            cnn_loss.append(cnn_metrics["loss"])

        progress.update(cnn_task, advance=1, description=f"Training CNN Model... Loss: {np.mean(cnn_loss):.4f} ± {np.std(cnn_loss):.4f}")

        # Compute mean/std for this epoch
        metrics_history['milo_loss'].append(np.mean(milo_loss))
        metrics_history['milo_loss_std'].append(np.std(milo_loss))
        metrics_history['milo_accuracy'].append(np.mean(milo_accuracy))
        metrics_history['milo_accuracy_std'].append(np.std(milo_accuracy))
        metrics_history['milo_time'].append(np.mean(milo_time))
        metrics_history['milo_time_std'].append(np.std(milo_time))

        metrics_history['cnn_loss'].append(np.mean(cnn_loss))
        metrics_history['cnn_loss_std'].append(np.std(cnn_loss))
        metrics_history['cnn_accuracy'].append(np.mean(cnn_accuracy))
        metrics_history['cnn_accuracy_std'].append(np.std(cnn_accuracy))
        metrics_history['cnn_time'].append(np.mean(cnn_time))
        metrics_history['cnn_time_std'].append(np.std(cnn_time))

        # Save best model by loss (corrected condition)
        if np.mean(milo_loss) < milo_best_loss:
            milo_best_loss = np.mean(milo_loss)
            milo_best_state = milo_state

        if np.mean(cnn_loss) < cnn_best_loss:
            cnn_best_loss = np.mean(cnn_loss)
            cnn_best_state = cnn_state


from flax.serialization import to_bytes
import json
import numpy as np

os.makedirs("models", exist_ok=True)

# --- Save model states ---
with open("models/milo_best_state.msgpack", "wb") as f:
    f.write(to_bytes(milo_best_state))

with open("models/cnn_best_state.msgpack", "wb") as f:
    f.write(to_bytes(cnn_best_state))

# --- Save metrics ---
# JSON version (for plotting)
def convert_metrics_to_serializable(metrics):
    return {k: list(map(float, v)) for k, v in metrics.items()}

with open("models/mnist_metrics_history.json", "w") as f:
    json.dump(convert_metrics_to_serializable(metrics_history), f)