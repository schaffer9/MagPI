__all__ = (
    'partial', 'jax', 'reduce', 'T', 'Any', 'Callable', 'Sequence', 'Optional',
    'f16', 'f32', 'f64', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'DeviceArray', 'array', 'zeros', 'zeros_like', 'ones', 'ones_like',
    'deg2rad', 'rad2deg', 'sum', 'exp', 'sin', 'cos', 'tan', 'hstack', 'vstack', 'stack', 'concatenate', 'eye', 'diag', 'dot', 'matmul', 'vdot',
    'grad', 'jit', 'jacfwd', 'jacrev', 'vmap', 'random', 'jnp', 'lax', 'jvp', 'flatten_util',
    'selu', 'elu', 'tanh', 'sigmoid', 'swish',
    'tree_map', 'tree_reduce', 'tree_leaves', 'flax', 'freeze', 'unfreeze', 'nn'
)

from functools import partial, reduce
import typing as T
from typing import Any, Callable, Sequence, Optional

import jax
from jax.numpy import (
    float16 as f16,
    float32 as f32,
    float64 as f64,
    int16 as i16,
    int32 as i32,
    int64 as i64,
    uint8 as u8,
    uint16 as u16,
    uint32 as u32,
    uint64 as u64,
    DeviceArray,
    array, zeros, zeros_like, ones, ones_like, eye, diag,
    deg2rad, rad2deg, sum, exp, sin, cos, tan,
    hstack, vstack, stack, concatenate, dot, matmul, vdot

)

import jax.numpy as jnp
from jax import (
    grad, jit, jacfwd, jacrev, vmap, random, lax, jvp, flatten_util
)

from jax.nn import (
    selu, elu, tanh, sigmoid, swish
)

from jax.tree_util import (
    tree_map, tree_reduce, tree_leaves
)

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
