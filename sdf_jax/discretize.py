import jax
import jax.numpy as jnp
from jax import vmap
from jax import lax
import functools as ft


def lmap(f):
    return lambda x: lax.map(f, x)

@ft.partial(jax.jit, static_argnums=(0,1))
def discretize2d(sdf, ngrid, x_lims, y_lims):
    xs = jnp.meshgrid(jnp.linspace(*x_lims, ngrid), 
                      jnp.linspace(*y_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, 2)
    ys = lmap(vmap(sdf))(xs)
    return xs, ys

@ft.partial(jax.jit, static_argnums=(0,1))
def discretize3d(sdf, ngrid, x_lims, y_lims, z_lims):
    xs = jnp.meshgrid(jnp.linspace(*x_lims, ngrid),
                      jnp.linspace(*y_lims, ngrid),
                      jnp.linspace(*z_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, nz, 3)
    ys = lmap(lmap(vmap(sdf)))(xs)
    return xs, ys
