import jax
import jax.numpy as jnp
from jax import vmap
import functools as ft


@ft.partial(jax.jit, static_argnums=(0,2))
def discretize2d(sdf, xy_lims, ngrid):
    xs = jnp.meshgrid(jnp.linspace(*xy_lims, ngrid), 
                      jnp.linspace(*xy_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, 2)
    ys = vmap(vmap(sdf))(xs)
    return xs, ys

@ft.partial(jax.jit, static_argnums=(0,2))
def discretize3d(sdf, xyz_lims, ngrid):
    xs = jnp.meshgrid(jnp.linspace(*xyz_lims, ngrid),
                      jnp.linspace(*xyz_lims, ngrid),
                      jnp.linspace(*xyz_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, nz, 3)
    ys = vmap(vmap(vmap(sdf)))(xs)
    return xs, ys
