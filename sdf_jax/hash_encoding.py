import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import functools as ft
from jaxtyping import Array, Float, UInt


PRNGKey = UInt[Array, "2"]


def hash_vertex(v: Float[Array, "3"], hashmap_size: UInt) -> UInt:
    primes = jnp.array([1, 2654435761, 805459861], dtype=np.uint32)
    h = np.uint32(0)
    for i in range(len(v)):
        h ^= v[i] * primes[i]
    return h % hashmap_size

def interpolate_bilinear(
    values:  Float[Array, "4 d"], 
    weights: Float[Array, "2"],
) -> Float[Array, "d"]:
    c0 = values[0]*(1.0-weights[0]) + values[2]*weights[0]
    c1 = values[1]*(1.0-weights[0]) + values[3]*weights[0]
    c = c0*(1.0-weights[1]) + c1*weights[1]
    return c

def interpolate_trilinear(
    values:  Float[Array, "8 d"], 
    weights: Float[Array, "3"],
) -> Float[Array, "d"]:
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
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
    if dim == 2: return np.array([[i,j] for i in (0,1) for j in (0,1)], dtype=np.uint32)
    elif dim == 3: return np.array([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)], dtype=np.uint32)
    else: assert False

@ft.partial(jax.jit, static_argnames=("nmin", "nmax"))
def encode(
    x:     Float[Array, "input_dim"], 
    theta: Float[Array, "levels hashmap_size features_per_entry"], 
    nmin:  UInt=16, 
    nmax:  UInt=512,
) -> Float[Array, "levels features_per_entry"]:
    """Multiresolution Hash Encoding.

    Following the paper:
        Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
        Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller
        ACM Transactions on Graphics (SIGGRAPH), July 2022

    The present code takes only a single input vector in 2D or 3D to encode.
    If you need to encode large batches of inputs jointly, consider
    wrapping this function with `jax.vmap`.
    """
    input_dim, = x.shape
    levels, hashmap_size, features_per_entry = theta.shape
    box = unit_box(input_dim)
    b = np.exp((np.log(nmax) - np.log(nmin)) / (levels - 1))
    def features(l):
        nl = jnp.floor(nmin * b**l)
        xl = x * nl
        xl_ = jnp.floor(xl).astype(np.uint32)

        # hash voxel vertices
        indices = jax.vmap(lambda v: hash_vertex(xl_ + v, hashmap_size))(box)

        # lookup
        tl = theta[l][indices]

        # interpolate
        wl = (xl - xl_)
        xi = interpolate_dlinear(tl, wl)

        return xi
    return jax.lax.map(features, np.arange(levels, dtype=np.uint32))

def init_encoding(
    key:                PRNGKey,
    levels:             UInt=16,
    hashmap_size_log2:  UInt=14,
    features_per_entry: UInt=2,
):
    hashmap_size = 1 << hashmap_size_log2
    theta = jrandom.uniform(key, (levels, hashmap_size, features_per_entry), minval=-0.0001, maxval=0.0001)
    return theta
