import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Tuple, Optional
from flax.linen import initializers
import jax.lax as lax
from flax.typing import (
    Dtype,
    Initializer,
)

default_weight_init = initializers.lecun_normal()

class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x += residual  # Residual connection
        x = nn.relu(x)
        return x


class MiloResidual(nn.Module):
    input_dim: Tuple[int, int]
    hidden_layer_dim: Sequence[Tuple[int, int]]
    output_dim: Tuple[int, int]

    @nn.compact
    def milo_branch(self, x):
        residual = x

        current_input_rows = self.input_dim[0]
        current_input_cols = self.input_dim[1]
        for i, (layer_left, layer_right) in enumerate(self.hidden_layer_dim):
            x = MiloNet(current_input_rows, current_input_cols, layer_left, layer_right)(x)

            if residual.shape != x.shape:
                residual = MiloNet(current_input_rows, current_input_cols, layer_left, layer_right)(residual)

            if i % 2 == 0: #Alternate residual connections
                x += residual

            x = nn.relu(x)

            current_input_rows = layer_left
            current_input_cols = layer_right

        x = MiloNet(current_input_rows, current_input_cols, self.output_dim[0], self.output_dim[1])(x)
        return x
    
    @nn.compact
    def __call__(self, x):
        x_out = self.milo_branch(x)
        return x_out


class MiloNet(nn.Module):
    input_rows: int
    input_cols: int
    next_dim_size_left: int
    next_dim_size_right: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    weight_init: Initializer = default_weight_init
    bias_init: Initializer = initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        weight_matrix_left = self.param('weight_left', self.weight_init, (self.next_dim_size_left, self.input_rows), self.param_dtype,)
        weight_matrix_right = self.param('weight_right', self.weight_init, (self.input_cols, self.next_dim_size_right), self.param_dtype,)      
        bias = self.param('bias', self.bias_init, (self.next_dim_size_left, self.next_dim_size_right), self.param_dtype)

        y = lax.dot_general(lhs=weight_matrix_left, rhs=inputs, dimension_numbers=(((1,), (1,)), ((), ())))
        y = lax.dot_general(lhs=weight_matrix_right, rhs=y, dimension_numbers=(((0,), (inputs.ndim - 1,)), ((), ()))).swapaxes(0, inputs.ndim - 1)
        y += jnp.reshape(bias, ( -1, bias.shape[0], bias.shape[1]))

        return y
    
    
class Milo(nn.Module):
    input_dim: Tuple[int, int]
    hidden_layer_dim: Sequence[Tuple[int, int]]
    output_dim: int 
    channel_output_dim: Optional[Tuple[int, int]] = None
    num_channels: Optional[int] = None

    @nn.compact
    def __call__(self, x):

        if self.num_channels == None:
            #Single channel does not need multiple milos
            model = MiloResidual(
                input_dim=self.input_dim,
                hidden_layer_dim=self.hidden_layer_dim,
                output_dim=self.output_dim
            )

            logits = model(x)
 
        else:
            # One MILO branch definition (parameters shared for all channels)
            milo_branch = MiloResidual(
                input_dim=self.input_dim,
                hidden_layer_dim=self.hidden_layer_dim,
                output_dim=self.channel_output_dim
            )

            channel_outputs = jax.vmap(milo_branch, in_axes=-1)(x)

            channel_outputs = channel_outputs.squeeze(-1)
            channel_outputs = jnp.moveaxis(channel_outputs, 0, 1) 
            combined = channel_outputs.reshape(channel_outputs.shape[0], -1) 

            # Final MLP for classification
            logits = nn.Dense(10)(combined)

        return logits


class CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        for _ in range(3):
            x = ResidualBlock(64)(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x
