import contextlib
import logging

import fiddle as fdl
import fiddle.extensions.jax
from absl import app
from etils import epath
from fiddle import absl_flags as fdl_flags
from fiddle import graphviz, printing
from fiddle.codegen import codegen
from fiddle.experimental import serialization

from src import configs

_RUNNER_CONFIG = fdl_flags.DEFINE_fiddle_config(
    "runner",
    default=None,
    help_string="The runner to use. If not specified, the default runner will be used.",
    default_module=configs,
)
_WORKDIR = epath.DEFINE_path(
    "workdir",
    default=None,
    help="The working directory.",
    required=True,
)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Unrecognized command line arguments: %s" % argv)

    config = _RUNNER_CONFIG.value

    # Print flattened representation
    logging.info(printing.as_str_flattened(config))

    _WORKDIR.value.mkdir(parents=True, exist_ok=True)
    # Write JSON representation
    (_WORKDIR.value / "config.json").write_text(serialization.dump_json(config))
    # Write Python representation
    (_WORKDIR.value / "config.py").write_text(
        codegen.new_codegen(config, top_level_fixture_name="base")
    )
    # Render graph representation
    graphviz.render(config).render(
        filename=_WORKDIR.value / "config.gv",
        outfile=_WORKDIR.value / "config.svg",
        format="svg",
    )

    runner = fdl.build(config)
    with contextlib.ExitStack() as stack:
        # Maybe enter some context managers while running
        runner.run(_WORKDIR.value)


if __name__ == "__main__":
    fiddle.extensions.jax.enable()
    app.run(main)
