from model import MLP
from train import create_state_MLP, train_MLP, eval_MLP
from utils import batch_loader
import numpy as np
import jax
import jax.numpy as jnp

#Dummy data
dataset_x = np.random.random((100, 256, 256))
dataset_y = np.random.randint(1, 40, 100)

#Create model

lr = 1e-3

rng = jax.random.PRNGKey(0)
device = jax.devices("cpu")[0] #Currently CPU

milo = MLP([128, 64], output_dim=1)
milo_state = create_state_MLP(rng, milo, lr, data_size=(100, 256, 256), device=device)


del rng

#Train model

NUM_EPOCHS = 100
EVAL_AFTER = 10
BATCH_SIZE = 32

best_loss = jnp.inf

metrics_history = {'train_loss': [], 'eval_loss': []}

for epoch in range(NUM_EPOCHS):
    train_losses = []

    for this_batch in batch_loader(dataset_x, dataset_y, BATCH_SIZE, device):
        state = train_MLP(milo_state, this_batch)
    
        train_metrics = eval_MLP(state, this_batch)
        train_losses.append(train_metrics["loss"])
    
    train_loss = sum(train_losses)/len(train_losses)

    if epoch % EVAL_AFTER == 0:
        # Logging
        with jax.disable_jit():
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}")