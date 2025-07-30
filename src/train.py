import jax.numpy as jnp
import optax
import jax
from flax.training import train_state

def create_state_MLP(rng, model, learning_rate, data_size, device=None):
    params = model.init(rng, jnp.ones(data_size))["params"]
    tx = optax.chain(
        optax.adam(learning_rate)
    )

    if device is not None:
        params = jax.device_put(params, device)

    return train_state.TrainState.create(apply_fn=model.apply, params = params, tx = tx)
