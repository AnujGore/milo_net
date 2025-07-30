import jax
import numpy as np
import jax.numpy as jnp
import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
from rich.progress import track
import optax

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # Moves one directory up
sys.path.append(project_root)

from src.model import MiloMLP
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
test_ds: tf.data.Dataset = tfds.load('cifar10', split='test')

train_ds = train_ds.map(preprocess).cache().shuffle(1024).batch(32, drop_remainder=True).take(2500).prefetch(1)
test_ds = test_ds.map(preprocess).batch(32, drop_remainder=True).prefetch(1)

#Hyperparameters
BATCH_SIZE = 32
lr = 1e-3
NUM_EPOCHS = 10

milo_multichannel = MiloMLP(
    input_dim=(32, 32), 
    hidden_layer_dim=[(24, 20), (14, 10), (10, 1)],
    output_dim=(10, 1),
    num_channels=3
)

milo_multichannel_state = create_state_MLP(rng, milo_multichannel, lr, data_size=(BATCH_SIZE, 32, 32, 3), device=device)

del rng

milo_multichannel_best_loss = np.inf; milo_multichannel_best_state = None

metrics_history = {
    'loss': [],
    'loss_std': [],
    'accuracy': [],
    'accuracy_std': []
}

for epoch in track(range(NUM_EPOCHS), description="Training CiFAR Model..."):
    milo_multichannel_loss = []
    milo_multichannel_accuracy = []
    for step, batch in enumerate(train_ds.as_numpy_iterator()):        
        labels = np.expand_dims(batch[1], axis = (1))
        milo_multichannel_state, this_loss, accuracy = train_step(milo_multichannel_state, batch)
        milo_multichannel_loss.append(this_loss)
        milo_multichannel_accuracy.append(accuracy)

    metrics_history['loss'].append(np.mean(milo_multichannel_loss))
    metrics_history['loss_std'].append(np.std(milo_multichannel_loss))
    metrics_history['accuracy'].append(np.mean(milo_multichannel_accuracy))
    metrics_history['accuracy_std'].append(np.std(milo_multichannel_accuracy))

    if np.mean(milo_multichannel_loss) < milo_multichannel_best_loss:
        milo_multichannel_best_loss = np.mean(milo_multichannel_loss)
        milo_multichannel_best_state = milo_multichannel_state


loss = metrics_history['loss']
loss_std = metrics_history['loss_std']

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]
    })

plt.figure(figsize=(10, 6))

plt.plot(loss, label='Loss', marker='o', linestyle='-', markersize=6)
plt.fill_between(range(len(loss)), np.array(loss) - np.array(loss_std), np.array(loss) + np.array(loss_std), alpha=0.2, label='Loss Std. Dev.')

plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


labels = np.random.randint(9, 18, 4)
labels = np.expand_dims(labels, axis = (1, 2))

fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # 1 row, 4 columns

img = eval_step(milo_multichannel_best_state, labels)  

for i in range(4):
    label = labels[i]
    axs[i].imshow(img[i])         # Display the image
    axs[i].axis('off')         # Hide axis
    axs[i].set_title(f'Label: {label[0, 0]}')  # Set the title to the label

plt.tight_layout()
plt.show()