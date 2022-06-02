import jax.numpy as jnp

def sdf_sphere(x, radius=0.3, center=jnp.array([0.5, 0.5])):
    return jnp.linalg.norm(x - center) - radius
