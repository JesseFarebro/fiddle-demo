from typing import Annotated

import fiddle as fdl
import flax.linen as nn
import jax.numpy as jnp


class WidthTag(fdl.Tag):
    """A tag for the width of a model."""


class DTypeTag(fdl.Tag):
    """A tag for a general dtype."""


class ComputeDTypeTag(fdl.Tag):
    """A tag for a dtype used for computation."""


class ParamDTypeTag(fdl.Tag):
    """A tag for a dtype used for parameters."""


class MyFancyModel(nn.Module):
    width: Annotated[int, WidthTag] = 1
    dtype: Annotated[jnp.dtype, DTypeTag, ComputeDTypeTag] = jnp.float32
    param_dtype: Annotated[jnp.dtype, DTypeTag, ParamDTypeTag] = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10 * self.width, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
