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

def interpolate_bilinear(values, weights):
    assert weights.shape == (2,)
    c0 = values[0]*(1.0-weights[0]) + values[2]*weights[0]
    c1 = values[1]*(1.0-weights[0]) + values[3]*weights[0]
    c = c0*(1.0-weights[1]) + c1*weights[1]
    return c

def interpolate_trilinear(values, weights):
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    assert weights.shape == (3,)
    c00 = values[0]*(1.0-weights[0]) + values[4]*weights[0]
    c01 = values[1]*(1.0-weights[0]) + values[5]*weights[0]
    c10 = values[2]*(1.0-weights[0]) + values[6]*weights[0]
    c11 = values[3]*(1.0-weights[0]) + values[7]*weights[0]
    c0 = c00*(1.0-weights[1]) + c10*weights[1]
    c1 = c01*(1.0-weights[1]) + c11*weights[1]
    c = c0*(1.0-weights[2]) + c1*weights[2]
    return c

def interpolate_dlinear(values, weights):
    dim, = weights.shape
    if dim == 2: return interpolate_bilinear(values, weights)
    elif dim == 3: return interpolate_trilinear(values, weights)
    else: assert False

def unit_box(dim: int):
    if dim == 2: return np.array([[i,j] for i in (0,1) for j in (0,1)], dtype=jnp.uint64)
    elif dim == 3: return np.array([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)], dtype=jnp.uint64)
    else: assert False

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
        xi = interpolate_dlinear(tl, wl)

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
