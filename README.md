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

The Multiresolution Hash Encoding in [sdf_jax/hash_encoding.py](sdf_jax/hash_encoding.py) implements the method described in
> __Instant Neural Graphics Primitives with a Multiresolution Hash Encoding__  
> [Thomas Müller](https://tom94.net), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _ACM Transactions on Graphics (__SIGGRAPH__), July 2022_  
> __[Website](https://nvlabs.github.io/instant-ngp/)&nbsp;/ [Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)&nbsp;/ [Code](https://github.com/NVlabs/instant-ngp)&nbsp;/ [Video](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4)&nbsp;/ [BibTeX](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.bib)__


## Installation
To ensure reproducibility, to install this repo and its dev dependencies: 
1. Use [Poetry](https://python-poetry.org/). 
    Make sure you have a local installation of Python `>=3.8` (e.g. by running `pyenv local 3.X.X`) and run:
    ```bash
    poetry install 
    ```

1. Alternatively, I've also included a `requirements.txt` that was generated from the `pyproject.toml` and `poetry.lock` files.
