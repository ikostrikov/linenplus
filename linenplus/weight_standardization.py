from typing import Any, Callable, Iterable, TypeVar, Union

import flax.linen as nn
import jax.numpy as jnp

T = TypeVar('T')
Array = Any

# From:
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py#L33-L65


def standardize(x: Array, axis: Union[int, Iterable[int]], eps: float):
    x = x - jnp.mean(x, axis=axis, keepdims=True)
    x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
    return x


class WSConv(nn.Conv):

    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
        param = super().param(name, init_fn, *init_args)
        if name == 'kernel':
            param = standardize(param, axis=[0, 1, 2], eps=1e-10)
        return param


class WSDense(nn.Dense):

    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
        param = super().param(name, init_fn, *init_args)
        if name == 'kernel':
            param = standardize(param, axis=0, eps=1e-10)
        return param
