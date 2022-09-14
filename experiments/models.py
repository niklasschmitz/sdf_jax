import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import treex as tx
from sdf_jax import hash_encoding
from typing import Callable, List, Sequence


class MLP(tx.Module):
    def __init__(self, dims: Sequence[int], act: Callable):
        self.dims = dims
        self.act = act
    @tx.compact
    def __call__(self, x):
        assert x.ndim == 1
        for dim in self.dims:
            x = tx.Linear(dim)(x)
            x = self.act(x)
        y = tx.Linear(1)(x)
        return y[0]

###############################################################
# Instant Neural Graphics Primitives with 
# a Multiresolution Hash Encoding (Müller et al, SIGGRAPH 22)
###############################################################

class HashEmbedding(tx.Module):
    theta: jnp.ndarray = tx.Parameter.node()

    def __init__(
        self, 
        levels: int=16, 
        hashmap_size_log2: int=14, 
        features_per_entry: int=2,
        nmin: int=16,
        nmax: int=512,
    ):
        self.levels = levels
        self.hashmap_size_log2 = hashmap_size_log2
        self.features_per_entry = features_per_entry
        self.nmin = nmin
        self.nmax = nmax

    def __call__(self, x):
        assert x.ndim == 1
        if self.initializing():
            hashmap_size = 1 << self.hashmap_size_log2
            key = tx.next_key()
            self.theta = jrandom.uniform(
                key, 
                (self.levels, hashmap_size, self.features_per_entry), 
                minval=-0.0001, 
                maxval=0.0001
            )
        
        y = hash_encoding.encode(x, self.theta, self.nmin, self.nmax)
        return y.reshape(-1)


def build_hash_mlp(emb_kwargs, hidden, act):
    model = tx.Sequential(
        HashEmbedding(**emb_kwargs),
        MLP([hidden, hidden], act),
    )
    return model

###############################################################
# Implicit Geometric Regularization for Learning Shapes
# (Gropp et al, ICML 2020)
###############################################################

def softplus(x, beta=100):
    return jnp.logaddexp(beta*x, 0) / beta

class IGRModel(tx.Module):
    hidden_layers: List[tx.Linear]
    final_layer: tx.Linear
    def __init__(
        self, 
        input_dim: int, 
        depth: int, 
        hidden: int, 
        act: Callable=softplus,
        radius_init=1.0,
    ):
        self.input_dim = input_dim
        self.depth = depth
        self.hidden = hidden
        self.act = act
        self.radius_init = radius_init

        hidden_layers = []
        for i in range(self.depth):

            # prepare skip connection one layer earlier
            if i == 2:
                h = self.hidden - self.input_dim 
            else: 
                h = self.hidden

            # geometric initialization
            layer = tx.Linear(
                h,
                kernel_init=jax.nn.initializers.normal(np.sqrt(2) / np.sqrt(h)),
                bias_init=jax.nn.initializers.constant(0.0),
            )
            hidden_layers += [layer]
        self.hidden_layers = hidden_layers
        
        def kernel_init(key, shape, dtype):
            mean = np.sqrt(np.pi) / np.sqrt(h)
            stddev = 0.00001
            return mean + stddev * jrandom.normal(key, shape, dtype)
        self.final_layer = tx.Linear(
            1, 
            kernel_init=kernel_init,
            bias_init=jax.nn.initializers.constant(-self.radius_init),
        )

    def __call__(self, x):
        assert x.ndim == 1

        # x is assumed in [0,1]^d
        # we now rescale to [-1,1]^d for geometric init to work
        x = 2*x - 1.0

        y = x
        for (i, layer) in enumerate(self.hidden_layers):
            # skip connection to the fourth layer
            if i + 1 == 4:
                y = jnp.concatenate([y, x])
            y = layer(y)
            y = self.act(y)
        y = self.final_layer(y)

        # rescaling values too to fit [0,1]^d instead of [-1,1]^d
        y = y / 2.0

        return y[0]


###############################################################
# Geometry-consistent Neural Shape Representation With
# Implicit Displacement Fields (Yifan et al, ICLR 2022)
###############################################################

def chi(fx, nu):
    return 1.0 / (1.0 + (fx / nu)**4)

class IDFModel(tx.Module):
    base: tx.Module = tx.node()
    disp: tx.Module = tx.node()

    def __init__(self, base, disp, nu: float):
        self.base = base
        self.disp = disp
        self.nu = nu

    def __call__(self, x):
        fx, gradfx = jax.value_and_grad(self.base)(x)
        x2 = x + chi(fx, self.nu) * self.disp(x) * gradfx / jnp.linalg.norm(gradfx)
        return self.base(x2)
