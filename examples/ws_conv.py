import jax
import jax.numpy as jnp
from jax._src.numpy.lax_numpy import var

from linenplus import WSConv

ws_conv = WSConv(32, (3, 3))

inputs = jnp.zeros((32, 8, 8, 3))

variables = ws_conv.init(jax.random.PRNGKey(42), inputs)

outputs = ws_conv.apply(variables, inputs)