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
