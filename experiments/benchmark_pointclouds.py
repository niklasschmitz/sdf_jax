import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from tqdm import tqdm
import functools as ft
import optax
import treex as tx
from pathlib import Path
import argparse
import pickle
import cloudpickle
from models import IGRModel, build_hash_mlp, IDFModel
from sdf_jax.util import dataloader
from sdf_jax.util import extract_mesh3d


def load_points(key, path):
    """Loads a dataset from https://github.com/tovacinni/sdf-explorer"""
    data_npz = jnp.load(path)
    data = {}
    # transform from [-1,1]^3 to [0,1]^3
    data["position"] = jnp.array(data_npz["position"]) / 2.0 + 0.5
    data["distance"] = jnp.array(data_npz["distance"]) / 2.0
    data["gradient"] = jnp.array(data_npz["gradient"]) / 2.0
    valid_indices = jnp.logical_not(jnp.isnan(
        data["position"].sum(axis=1)
        + data["distance"].reshape(-1)
        + data["gradient"].sum(axis=1)
    ))
    data["position"] = data["position"][valid_indices]
    data["distance"] = data["distance"][valid_indices].squeeze()
    data["gradient"] = data["gradient"][valid_indices]
    indices = jnp.arange(len(data["position"]))
    shuffled_indices = jrandom.permutation(key, indices)
    n_train = len(shuffled_indices) // 10
    train_indices, test_indices = shuffled_indices[:n_train], shuffled_indices[n_train:]
    data_train = jax.tree_map(lambda x: x[train_indices], data)
    data_test = jax.tree_map(lambda x: x[test_indices], data)
    return data_train, data_test


# IGR loss

def sample_normal_per_point(key, xs, local_sigma=0.01):
    key, key_local, key_global = jrandom.split(key, 3)
    sample_local = xs + jrandom.normal(key_local, xs.shape) * local_sigma
    sample_global = jrandom.uniform(key_global, (xs.shape[0], xs.shape[1]))
    return jnp.vstack([sample_local, sample_global])

def surface_loss_fn(model, x):
    return jnp.abs(model(x))

def normal_loss_fn(model, x, normal):
    return jnp.linalg.norm(jax.grad(model)(x) - normal)

def eikonal_loss_fn(model, x):
    return (jnp.linalg.norm(jax.grad(model)(x)) - 1.0)**2

@ft.partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, xs, normals, lam, tau, key):
    model = model.merge(params)
    surface_loss = jnp.mean(jax.vmap(ft.partial(surface_loss_fn, model))(xs))
    if normals is not None:
        normal_loss = jnp.mean(jax.vmap(ft.partial(normal_loss_fn, model))(xs, normals))
    else:
        normal_loss = 0.0
    xs_eik = sample_normal_per_point(key, xs)
    eikonal_loss = jnp.mean(jax.vmap(ft.partial(eikonal_loss_fn, model))(xs_eik))
    loss = surface_loss + tau * normal_loss + lam * eikonal_loss
    return loss, model

@jax.jit
def train_step(model, xs, normals, lam, tau, optimizer, key):
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, xs, normals, lam, tau, key)
    new_params = optimizer.update(grads, params)
    model = model.merge(new_params)
    return loss, model, optimizer

@jax.jit
def test_step(model, xs, normals):
    surface_loss = jnp.mean(jax.vmap(ft.partial(surface_loss_fn, model))(xs))
    if normals is not None:
        normal_loss = jnp.mean(jax.vmap(ft.partial(normal_loss_fn, model))(xs, normals))
    else:
        normal_loss = 0.0
    xs_eik = sample_normal_per_point(key, xs)
    eikonal_loss = jnp.mean(jax.vmap(ft.partial(eikonal_loss_fn, model))(xs_eik))
    return surface_loss, normal_loss, eikonal_loss

# 

def save_model(modelpath, model):
    with open(modelpath, "wb") as modelfile:
        cloudpickle.dump(model, modelfile)

def load_model(modelpath):
    model_bytes = modelpath.read_bytes()
    model = pickle.loads(model_bytes)
    return model

def print_callback(step, loss, model, optimizer, finalize):
    tqdm.write(f"[{step}] loss: {loss:.8f}")

def fit(
    model,
    xs,
    normals=None,
    lam=0.1,
    tau=1.0,
    # optimizer
    key=jrandom.PRNGKey(1234),
    lr=5e-3,
    steps=100,
    batch_size=128,
    # utils
    cb=print_callback,
    cb_every=10,
):
    optimizer = tx.Optimizer(optax.adam(lr))
    optimizer = optimizer.init(model.filter(tx.Parameter))
    key, data_key = jrandom.split(key, 2)
    for step, (xs_batch, normals_batch) in zip(tqdm(range(steps)), dataloader(xs, normals, batch_size, key=data_key)):
        key, step_key = jrandom.split(key, 2)
        loss, model, optimizer = train_step(model, xs_batch, normals_batch, lam, tau, optimizer, step_key)
        if step % cb_every == 0:
            cb(step, loss, model, optimizer, finalize=False)
    cb(step, loss, model, optimizer, finalize=True)
    return loss, model

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='GDML-JAX MD17 Example')
    parser.add_argument('--data', type=str, default="Dalek.npz")
    parser.add_argument('--model', type=str, default="igr")
    parser.add_argument('--steps', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=128**2)
    parser.add_argument('--cb_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--save_path', type=str, default="checkpoints")
    parser.add_argument('--savegrid', type=int, default=256)
    args = parser.parse_args()

    key = jrandom.PRNGKey(1234)
    key, data_key, model_key, train_key = jrandom.split(key, 4)

    data_train, data_test = load_points(data_key, args.data)
    data_test = jax.tree_map(lambda y: y[:args.batch_size], data_test)

    modules = {
        "igr": IGRModel(input_dim=3, depth=7, hidden=512),
        "hash": build_hash_mlp(emb_kwargs={}, hidden=64, act=jax.nn.relu),
        "idf": IDFModel(
            IGRModel(input_dim=3, depth=7, hidden=512),
            build_hash_mlp(emb_kwargs={}, hidden=64, act=jax.nn.relu),
            nu=0.04,
        ),
    }
    module = modules[args.model]

    model = module.init(key=model_key, inputs=data_train["position"][0])

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    def save_callback(step, loss, model, optimizer, finalize):
        print_callback(step, loss, model, optimizer, finalize)
        if step % args.save_every == 0 or finalize:
            vertices, faces = extract_mesh3d(
                model, ngrid=args.savegrid, x_lims=(0,1), y_lims=(0,1), z_lims=(0,1),
            )
            surface_loss, normal_loss, eikonal_loss = test_step(
                model, data_test["position"], data_test["gradient"],
            )
            meta = {
                "loss": loss.item(), 
                "test_surface_loss": surface_loss.item(),
                "test_normal_loss": normal_loss.item(),
                "test_eikonal_loss": eikonal_loss.item(),
            }
            jnp.savez(save_path / f"mesh{step}.npz", vertices=vertices, faces=faces, meta=meta)
            tqdm.write(str(meta))

    loss, model = fit(
        model,
        xs=data_train["position"],
        normals=data_train["gradient"],
        steps=args.steps,
        batch_size=args.batch_size,
        cb_every=args.cb_every,
        cb=save_callback,
        key=train_key,
    )
