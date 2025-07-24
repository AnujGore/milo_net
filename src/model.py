import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from flax.linen import initializers
import jax.lax as lax
from flax.typing import (
    Dtype,
    Initializer,
)

default_weight_init = initializers.lecun_normal()

class MiloNet(nn.Module):
  next_dim_size: int
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  weight_init: Initializer = default_weight_init
  bias_init: Initializer = initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    weight_matrix_left = self.param('weight_left', self.weight_init, (self.next_dim_size, jnp.shape(inputs)[-1]), self.param_dtype,)
    weight_matrix_right = self.param('weight_right', self.weight_init, (jnp.shape(inputs)[-1], self.next_dim_size), self.param_dtype,)      
    bias = self.param('bias', self.bias_init, (self.next_dim_size, self.next_dim_size), self.param_dtype)

    y = lax.dot_general(lhs=weight_matrix_left, rhs=inputs, dimension_numbers=(((1,), (1,)), ((), ()))).transpose(1, 0, 2)
    y = lax.dot_general(lhs=weight_matrix_right, rhs=y, dimension_numbers=(((0,), (2,)), ((), ()))).transpose(1, 0, 2)
    y += jnp.reshape(bias, (-1, self.next_dim_size, self.next_dim_size))

    return y
  

class MLP(nn.Module):
    hidden_layer_dim: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for layer in self.hidden_layer_dim:
            x = MiloNet(layer)(x)
            x = nn.relu(x)        
        x = MiloNet(self.output_dim)(x)
    
        return x
    