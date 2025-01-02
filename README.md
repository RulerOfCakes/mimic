# Mimic

Mimic is a project inspired by [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html), recreating a similar
physics-based character model control system with Deep Reinforcement Learning.

# Dependencies

- [`Rapier`](https://rapier.rs/)
- [`Bevy`](https://bevyengine.org/)
- [`nalgebra`](https://nalgebra.org/)
- [`xurdf`](https://github.com/neka-nat/xurdf)
- [`pyo3`](https://pyo3.rs/)
- [`pytorch`](https://pytorch.org/)
- Python 3.12 installed on your system with shared libraries.
    - use the environment variable `PYO3_PYTHON` to specify your desired python environment's executable path.
    - The path to the shared library(`.dll` or `.so`) should also be in your `PATH`.
    - `pytorch` should be installed in that python environment.