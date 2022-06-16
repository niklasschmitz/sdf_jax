from sdf_jax.discretize import discretize2d, discretize3d

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure


def plot2d(sdf, xy_lims=(0, 1), ngrid=10, scatter_pts=None):
    xs, ys = discretize2d(sdf, xy_lims, ngrid)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.set_title("zero levelset")
    ax0.contour(xs[:,:,0], xs[:,:,1], ys, levels=[0.], colors=["salmon"])
    ax0.set_xticks(xy_lims)
    ax0.set_yticks(xy_lims)
    ax1.set_title("SDF")
    # c = ax1.imshow(ys, origin='lower')
    c = ax1.contourf(xs[:,:,0], xs[:,:,1], ys, levels=np.arange(-1,1,0.1))
    ax1.contour(xs[:,:,0], xs[:,:,1], ys, levels=[0.], colors=["w"])
    if scatter_pts is not None:
        ax0.scatter(*scatter_pts.T)
        ax1.scatter(*scatter_pts.T, c="w", alpha=0.5)
    ax1.set_xticks(xy_lims)
    ax1.set_yticks(xy_lims)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(c, cax=cax)
    cb.ax.plot([0, 1], [0.0]*2, c="w")
    plt.tight_layout()
    plt.close()
    return fig

def plot3d(sdf, xyz_lims=(0, 1), ngrid=10):
    xs, ys = discretize3d(sdf, xyz_lims, ngrid)
    verts, faces, normals, values = measure.marching_cubes(np.array(ys), 0)
    verts = (verts / ngrid) * abs(xyz_lims[0] - xyz_lims[1]) + xyz_lims[0]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False).show()

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
