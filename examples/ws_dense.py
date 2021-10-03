import jax
import jax.numpy as jnp

from linenplus import WSDense

ws_dense = WSDense(32)

inputs = jnp.zeros((32, 3))

variables = ws_dense.init(jax.random.PRNGKey(42), inputs)

outputs = ws_dense.apply(variables, inputs)