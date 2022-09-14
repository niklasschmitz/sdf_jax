import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from tqdm import tqdm
import functools as ft
import optax
import treex as tx
from sdf_jax.util import dataloader
from models import IGRModel
from pathlib import Path
import pickle
import cloudpickle


def load_points(path):
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
    data["distance"] = data["distance"][valid_indices]
    data["gradient"] = data["gradient"][valid_indices]
    return data

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

# 

def save_model(modelpath, model):
    with open(modelpath, "wb") as modelfile:
        cloudpickle.dump(model, modelfile)

def load_model(modelpath):
    model_bytes = modelpath.read_bytes()
    model = pickle.loads(model_bytes)
    return model

def print_callback(step, loss, model, optimizer):
    tqdm.write(f"[{step}] loss: {loss:.8f}")

def fit(
    module,
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
    key, model_key = jrandom.split(key, 2)
    model = module.init(key=model_key, inputs=xs[0])
    optimizer = tx.Optimizer(optax.adam(lr))
    optimizer = optimizer.init(model.filter(tx.Parameter))
    key, data_key = jrandom.split(key, 2)
    for step, (xs_batch, normals_batch) in zip(tqdm(range(steps)), dataloader(xs, normals, batch_size, key=data_key)):
        key, step_key = jrandom.split(key, 2)
        loss, model, optimizer = train_step(model, xs_batch, normals_batch, lam, tau, optimizer, step_key)
        if step % cb_every == 0:
            cb(step, loss, model, optimizer)
    cb(step, loss, model, optimizer)
    return loss, model

if __name__=='__main__':

    data = load_points("../../samples/surface/Dalek.npz")

    module = IGRModel(input_dim=3, depth=7, hidden=512)

    model = module.init(key=jrandom.PRNGKey(0), inputs=data["position"][0])

    save_every = 2000
    save_path = Path("checkpoints")
    save_path.mkdir(parents=True, exist_ok=True)

    def save_callback(step, loss, model, optimizer):
        print_callback(step, loss, model, optimizer)
        if step % save_every == 0:
            modelpath = save_path / f"model{step}.cpkt"
            save_model(modelpath, model)

    loss, model = fit(
        module,
        xs=data["position"],
        normals=data["gradient"],
        steps=10_000,       # 100_000
        batch_size=128**2, # 128**2
        cb_every=50,
        cb=save_callback,
    )
