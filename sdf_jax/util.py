from sdf_jax.discretize import discretize2d, discretize3d

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure


def plot2d(sdf, *, ngrid=10, x_lims=(0, 1), y_lims=None, scatter_pts=None):
    if y_lims is None:
        y_lims = x_lims
    xs, ys = discretize2d(sdf, ngrid, x_lims, y_lims)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.set_title("zero levelset")
    ax0.contour(xs[:,:,0], xs[:,:,1], ys, levels=[0.], colors=["salmon"])
    ax0.set_xticks(x_lims)
    ax0.set_yticks(y_lims)
    ax1.set_title("SDF")
    c = ax1.contourf(xs[:,:,0], xs[:,:,1], ys, levels=np.arange(-1,1,0.1))
    ax1.contour(xs[:,:,0], xs[:,:,1], ys, levels=[0.], colors=["w"])
    if scatter_pts is not None:
        ax0.scatter(*scatter_pts.T)
        ax1.scatter(*scatter_pts.T, c="w", alpha=0.5)
    ax1.set_xticks(x_lims)
    ax1.set_yticks(y_lims)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(c, cax=cax)
    cb.ax.plot([0, 1], [0.0]*2, c="w")
    plt.tight_layout()
    plt.close()
    return fig

def plot3d(sdf, *, ngrid=10, x_lims=(0, 1), y_lims=None, z_lims=None):
    if y_lims is None:
        y_lims = x_lims
    if z_lims is None:
        z_lims = x_lims
    xs, ys = discretize3d(sdf, ngrid, x_lims, y_lims, z_lims)
    verts, faces, normals, values = measure.marching_cubes(np.array(ys), 0)
    verts /= ngrid
    verts[:,0] = verts[:,0] * abs(x_lims[0] - x_lims[1]) + x_lims[0]
    verts[:,1] = verts[:,1] * abs(y_lims[0] - y_lims[1]) + y_lims[0]
    verts[:,2] = verts[:,2] * abs(z_lims[0] - z_lims[1]) + z_lims[0]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False).show()

def dataloader(xs, ys, batch_size, *, key):
    dataset_size = xs.shape[0]
    indices = jnp.arange(dataset_size)
    assert batch_size <= dataset_size
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            if ys is not None:
                yield xs[batch_perm], ys[batch_perm]
            else:
                yield xs[batch_perm], None
            start = end
            end = start + batch_size
