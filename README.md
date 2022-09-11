# sdf_jax

Utilities for neural signed distance fields in JAX.

## Content

```
sdf_jax
    ├── discretize.py     # utils for dense 2D and 3D grid evaluation of a field
    ├── examples.py       # for debugging: simple analytical SDFs like the sphere
    ├── hash_encoding.py  # Multiresolution Hash Encoding
    └── util.py           # plotting utils for level-sets from marching cubes
```

## Installation
To ensure reproducibility, to install this repo and its dev dependencies: 
1. Use [Poetry](https://python-poetry.org/). 
    Make sure you have a local installation of Python `>=3.8` (e.g. by running `pyenv local 3.X.X`) and run:
    ```bash
    poetry install 
    ```

1. Alternatively, I've also included a `requirements.txt` that was generated from the `pyproject.toml` and `poetry.lock` files.
