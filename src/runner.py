import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from etils import epath
from flax.training import train_state


@dataclasses.dataclass(kw_only=True)
class Runner:
    rng: jax.Array

    model: nn.Module
    optim: optax.GradientTransformation

    state: train_state.TrainState = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.rng, params_rng = jax.random.split(self.rng)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.lazy_init(
                params_rng, jax.ShapeDtypeStruct((128, 128, 3), jnp.float32)
            ),
            tx=self.optim,
        )

    def run(self, workdir: epath.Path) -> None:
        del workdir
        print("Tada!")
