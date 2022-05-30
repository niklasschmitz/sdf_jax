from typing import Callable, Sequence
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
import treex as tx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def sdf_sphere(x):
    return jnp.dot(x, x) - 1


@ft.partial(jax.jit, static_argnums=(0,2))
def discretize2d(sdf, xy_lims, ngrid):
    xs = jnp.meshgrid(jnp.linspace(*xy_lims, ngrid), 
                      jnp.linspace(*xy_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, 2)
    ys = vmap(vmap(sdf))(xs)
    return xs, ys

def plot2d(sdf, xy_lims=(0,1), ngrid=10):
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


@ft.partial(jax.jit, static_argnums=(0,2))
def discretize3d(sdf, xyz_lims, ngrid):
    xs = jnp.meshgrid(jnp.linspace(*xyz_lims, ngrid),
                      jnp.linspace(*xyz_lims, ngrid),
                      jnp.linspace(*xyz_lims, ngrid))
    xs = jnp.stack(xs, axis=-1) # (nx, ny, nz, 3)
    ys = vmap(vmap(vmap(sdf)))(xs)
    return xs, ys

def plot3d(sdf, xyz_lims=(0, 1), ngrid=10):
    xs, ys = discretize3d(sdf, xyz_lims, ngrid)
    verts, faces, normals, values = measure.marching_cubes(np.array(ys), 0)
    verts = (verts / ngrid) * abs(xyz_lims[0] - xyz_lims[1]) + xyz_lims[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(*xyz_lims)
    ax.set_ylim(*xyz_lims)
    ax.set_zlim(*xyz_lims)
    plt.tight_layout()


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

class SimpleNeuralSDF(tx.Module):
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
    
@ft.partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.merge(params)
    preds = jax.vmap(model)(x)
    # loss = 0.5 * jnp.mean((preds - y) ** 2)
    loss = jnp.mean(jnp.abs(preds - y) / (jnp.abs(y) + 0.01))
    return loss, model

@jax.jit
def train_step(model, x, y, optimizer):
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x, y)
    new_params = optimizer.update(grads, params)
    model = model.merge(new_params)
    return loss, model, optimizer

def print_callback(step, loss, model, optimizer):
    print(f"[{step}] loss: {loss:.4f}")
