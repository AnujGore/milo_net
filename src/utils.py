import jax.numpy as jnp
import jax
import tensorflow as tf

def batch_loader(dataset_x, dataset_y, batch_size, device):
    dataset_size = len(dataset_x)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = start_idx + batch_size
        batch_x = jnp.asarray(dataset_x[start_idx:end_idx])
        batch_y = jnp.asarray(dataset_y[start_idx:end_idx])
        # Put batch on device (GPU/TPU)
        batch = {
            "x": jax.device_put(batch_x, device),
            "y": jax.device_put(batch_y, device)
        }
        yield batch

def preprocess(sample):
    image = tf.cast(sample['image'], tf.float32) / 255.0
    return image, sample['label']