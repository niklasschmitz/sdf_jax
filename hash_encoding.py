import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.config import config
config.update("jax_enable_x64", True)  # TODO how to embrace int32 overflow for hashing in JAX?
import numpy as np

def hash_vertex(v, hashmap_size):
    # TODO how to embrace int32 overflow in JAX?
    primes = [1, 2654435761, 805459861]
    h = 0
    for i in range(len(v)):
        h ^= v[i] * primes[i]
    return h % hashmap_size

def trilinear_interpolation(values, weights):
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    c00 = values[int('000',2)]*(1.0-weights[0]) + values[int('100',2)]*weights[0]
    c01 = values[int('001',2)]*(1.0-weights[0]) + values[int('101',2)]*weights[0]
    c10 = values[int('010',2)]*(1.0-weights[0]) + values[int('110',2)]*weights[0]
    c11 = values[int('011',2)]*(1.0-weights[0]) + values[int('111',2)]*weights[0]
    c0 = c00*(1.0-weights[1]) + c10*weights[1]
    c1 = c01*(1.0-weights[1]) + c11*weights[1]
    c = c0*(1.0-weights[2]) + c1*weights[2]
    return c

def unit_box(dim: int):
    if dim == 2:
        return np.array([[i,j] for i in (0,1) for j in (0,1)], dtype=jnp.uint64)
    elif dim == 3:
        return np.array([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)], dtype=jnp.uint64)
    else:
        assert False

@jax.jit
def encode(x, theta, nmin=16, nmax=512):
    assert x.ndim == 1
    levels, hashmap_size, features_per_entry = theta.shape
    box = unit_box(x.shape[0])
    b = np.exp((np.log(nmax) - np.log(nmin)) / (levels - 1))
    def features(l):
        nl = jnp.floor(nmin * b**l)
        xl = x * nl
        xl_ = jnp.floor(xl).astype(jnp.uint64)

        # hash voxel vertices
        indices = jax.vmap(lambda v: hash_vertex(xl_ + v, hashmap_size))(box)

        # lookup
        tl = theta[l][indices]

        # interpolate
        wl = (xl - xl_)
        xi = trilinear_interpolation(tl, wl)

        return xi
    return jax.lax.map(features, np.arange(levels, dtype=jnp.uint64))

def init_encoding(
    key,
    levels: int=16,
    hashmap_size_log2: int=14,
    features_per_entry: int=2,
):
    hashmap_size = 1 << hashmap_size_log2
    theta = jrandom.uniform(key, (levels, hashmap_size, features_per_entry), minval=-0.0001, maxval=0.0001)
    return theta
