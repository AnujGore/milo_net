import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Tuple
from flax.linen import initializers
import jax.lax as lax
from flax.typing import (
    Dtype,
    Initializer,
)

default_weight_init = initializers.lecun_normal()

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

        y = lax.dot_general(lhs=weight_matrix_left, rhs=inputs, dimension_numbers=(((1,), (1,)), ((), ())))#.transpose(1, 0, 2)
        y = lax.dot_general(lhs=weight_matrix_right, rhs=y, dimension_numbers=(((0,), (2,)), ((), ())))#.transpose(1, 2, 0)
        y = y.transpose(2, 1, 0)
        y += jnp.reshape(bias, (-1, self.next_dim_size_left, self.next_dim_size_right))

        return y
    

class MiloMLP(nn.Module):
    input_dim: Tuple[int, int]
    hidden_layer_dim: Sequence[Tuple[int, int]]
    output_dim: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        current_input_rows = self.input_dim[0]
        current_input_cols = self.input_dim[1]
        for layer_left, layer_right in self.hidden_layer_dim:
            x = MiloNet(current_input_rows, current_input_cols, layer_left, layer_right)(x)
            current_input_rows = layer_left
            current_input_cols = layer_right
            x = nn.relu(x)
        x = MiloNet(current_input_rows, current_input_cols, self.output_dim[0], self.output_dim[1])(x)
        return x
    