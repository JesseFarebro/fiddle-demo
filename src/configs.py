import fiddle as fdl
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from fiddle import selectors
from fiddle.experimental import auto_config

from . import models, runner

# === Tags ===


class LearningRate(fdl.Tag):
    """A tag for a learning rate."""


# === Base configurations ===


@auto_config.auto_config
def base(seed: int = 0) -> runner.Runner:
    """The base configuration for our runner."""
    return runner.Runner(
        rng=jax.random.key(seed),
        model=models.MyFancyModel(
            width=1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        ),
        optim=optax.adam(
            learning_rate=auto_config.with_tags(1e-3, LearningRate),
        ),
    )


# === Fiddlers ===


def simple_model(cfg: fdl.Config[runner.Runner]) -> None:
    """Replace the model with a simple model."""
    cfg.model = fdl.Config(
        nn.Dense,
        features=10,
        dtype=auto_config.with_tags(jnp.float32, (models.DTypeTag, models.ComputeDTypeTag)),
        param_dtype=auto_config.with_tags(jnp.float32, (models.DTypeTag, models.ParamDTypeTag)),
    )


def cosine_lr_schedule(
    cfg: fdl.Config[runner.Runner],
    *,
    init_value: float,
    decay_steps: float,
) -> None:
    """Replace the learning rate schedule with a cosine decay schedule."""
    selectors.select(cfg, tag=LearningRate, check_nonempty=True).replace(
        fdl.Config(
            optax.cosine_decay_schedule,
            init_value=init_value,
            decay_steps=decay_steps,
        )
    )


def adamw(cfg: fdl.Config[runner.Runner]) -> None:
    """Replace the optimizer with AdamW."""
    selectors.select(cfg, optax.GradientTransformation).replace(
        fdl.Config(optax.adamw, learning_rate=LearningRate.new(1e-3))
    )


def half_precision_compute(cfg: fdl.Config[runner.Runner]) -> None:
    """Replace the computation dtype with bfloat16."""
    selectors.select(cfg, tag=models.ComputeDTypeTag).replace(jnp.bfloat16)


def half_precision_params(cfg: fdl.Config[runner.Runner]) -> None:
    """Replace the parameter dtype with bfloat16."""
    selectors.select(cfg, tag=models.ParamDTypeTag).replace(jnp.bfloat16)


def half_precision(cfg: fdl.Config[runner.Runner]) -> None:
    """Replace all dtypes with bfloat16."""
    selectors.select(cfg, tag=models.DTypeTag).replace(jnp.bfloat16)
