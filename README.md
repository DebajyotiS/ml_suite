<div align="center">

<img src="assets/ml_suite_rainbow.png" alt="ml_suite" width="480">

<br/>
<br/>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.12%2B-EE4C2C?logo=pytorch&logoColor=white"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
<a href="https://github.com/DebajyotiS/ml_suite/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/DebajyotiS/ml_suite/actions/workflows/tests.yml/badge.svg"></a>
<a href="https://github.com/DebajyotiS/ml_suite/actions/workflows/build.yml"><img alt="Build" src="https://github.com/DebajyotiS/ml_suite/actions/workflows/build.yml/badge.svg"></a>
<a href="https://github.com/DebajyotiS/ml_suite/actions/workflows/tests.yml"><img alt="Coverage" src="https://raw.githubusercontent.com/DebajyotiS/ml_suite/coverage-data/badge.svg"></a>
<a href="https://DebajyotiS.github.io/ml_suite/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white"></a>

<br/>
<br/>

<p>Reusable, dimension-agnostic deep learning building blocks for generative modelling research. Composable PyTorch primitives that work identically across 1D, 2D, and 3D data.</p>

</div>

---

## Installation

```bash
uv sync --extra dev   # recommended
pip install -e ".[dev]"  # pip alternative
```

Full documentation, API reference, and usage examples are **[here](https://DebajyotiS.github.io/ml_suite/)**.

---

## Development

```bash
pytest
ruff check src
ruff format --check src
```
