from sdf_jax.discretize import discretize2d, discretize3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def plot2d(sdf, xy_lims=(0,1), ngrid=10):
    xs, ys = discretize2d(sdf, xy_lims, ngrid)
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
