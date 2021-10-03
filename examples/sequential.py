import flax.linen as nn
import jax
import jax.numpy as jnp

from linenplus import Sequential

mlp = Sequential([
    nn.Dense(features=256),
    nn.relu,
    nn.Dense(features=256),
    nn.relu,
    nn.Dense(features=10),
])

inputs = jnp.zeros((32, 10))
variables = mlp.init(jax.random.PRNGKey(42), inputs)['params']
outputs = mlp.apply({'params': variables}, inputs)