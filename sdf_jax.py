from typing import Callable, Sequence
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
import equinox as eqx
from equinox import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def sdf_sphere(x):
    return jnp.dot(x, x) - 1


@eqx.filter_jit
def discretize2d(sdf, xy_lims, ngrid):
    xs = jnp.meshgrid(jnp.linspace(*xy_lims, ngrid), 
                      jnp.linspace(*xy_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, 2)
    ys = vmap(vmap(sdf))(xs)
    return xs, ys

def plot2d(sdf, xy_lims=(-2,2), ngrid=10):
    xs, ys = discretize2d(sdf, xy_lims, ngrid)
    # fig, axs = plt.subplots(ncols=2, figsize=(6,3))
    fig = plt.figure(figsize=(6.25,3))
    gs = plt.GridSpec(1,3, width_ratios=[3,3,0.25])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    c = ax0.imshow(ys, origin='lower')
    ax0.set_title("SDF")
    ax1.contour(ys, levels=[0.])
    ax1.set_title("zero levelset")
    for ax in [ax0, ax1]:
        ax.set_xticks([0,ngrid-1])
        ax.set_xticklabels(xy_lims)
        ax.set_yticks([0,ngrid-1])
        ax.set_yticklabels(xy_lims)
    plt.colorbar(c, cax=ax2)
    plt.tight_layout()



@eqx.filter_jit
def discretize3d(sdf, xyz_lims, ngrid):
    xs = jnp.meshgrid(jnp.linspace(*xyz_lims, ngrid),
                      jnp.linspace(*xyz_lims, ngrid),
                      jnp.linspace(*xyz_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, nz, 3)
    ys = vmap(vmap(vmap(sdf)))(xs)
    return xs, ys

def plot3d(sdf, xyz_lims=(-2,2), ngrid=10):
    xs, ys = discretize3d(sdf, xyz_lims, ngrid)
    verts, faces, normals, values = measure.marching_cubes(np.array(ys), 0)
    verts = (verts / 10) * 4 - 2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(*xyz_lims)
    ax.set_ylim(*xyz_lims)
    ax.set_zlim(*xyz_lims)
    plt.tight_layout()


class NeuralSDF(eqx.Module):
    layers: Sequence[nn.Linear]
    activation: Callable

    def __init__(
        self, 
        in_size: int, 
        width_size: int,
        depth: int,
        *,
        activation: Callable=jax.nn.relu,
        key: jrandom.PRNGKey,
    ):
        keys = jrandom.split(key, depth + 1)
        layers = [nn.Linear(in_size, width_size, key=keys[0])]
        for i in range(depth - 1):
            layers += [nn.Linear(width_size, width_size, key=keys[i+1])]
        layers += [nn.Linear(width_size, 1, key=keys[-1])]
        self.layers = layers
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        y = self.layers[-1](x)
        return y[0]

def dataloader(xs, ys, batch_size, *, key):
    dataset_size = xs.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield xs[batch_perm], ys[batch_perm]
            start = end
            end = start + batch_size

def single_loss_fn(model, x, y):
    return 0.5 * jnp.square(y - model(x))

def batch_loss_fn(model, xs, ys):
    loss_fn = ft.partial(single_loss_fn, model)
    return jnp.mean(vmap(loss_fn)(xs, ys))

@eqx.filter_jit
def make_step(model, xs, ys, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, xs, ys)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
