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

@jax.jit
def train_MLP(state, batch):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch["x"])
        preds = jnp.squeeze(preds)
        loss = optax.l2_loss(preds, batch["y"]).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
   
    state = state.apply_gradients(grads = grads)

    return state

@jax.jit
def eval_MLP(state, batch):
    preds = state.apply_fn({"params": state.params}, batch["x"])
    preds = jnp.squeeze(preds)
    loss = optax.l2_loss(preds, batch["y"]).mean()
    return {"loss": loss,
            "preds": preds,
            "target": batch["y"]}
