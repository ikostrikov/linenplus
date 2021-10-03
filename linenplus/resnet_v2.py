# Based on:
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# and
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ResNetV2Block(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.norm()(x)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides)(residual)

        return residual + y


class CifarResNetV2(nn.Module):
    """ResNetV2."""
    stage_sizes: Sequence[int]
    num_classes: int
    conv_cls: ModuleDef
    norm_cls: ModuleDef
    num_filters: int = 16
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv_cls, use_bias=False, dtype=self.dtype)
        norm = partial(self.norm_cls,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        x = conv(self.num_filters, (3, 3))(x)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetV2Block(self.num_filters * 2**i,
                                  strides=strides,
                                  conv=conv,
                                  norm=norm,
                                  act=self.act)(x)
        x = norm()(x)
        x = self.act(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x
