import argparse
import logging
from pathlib import Path
import pickle
import cloudpickle

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import functools as ft
import optax
import treex as tx
from models import IGRModel, build_hash_mlp, IDFModel
from sdf_jax.util import extract_mesh3d


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

def dataloader(xs, ys, normals, batch_size, *, key):
    dataset_size = xs.shape[0]
    assert batch_size <= dataset_size
    assert dataset_size == ys.shape[0] == normals.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield xs[batch_perm], ys[batch_perm], normals[batch_perm]
            start = end
            end = start + batch_size

# dense supervised loss

@ft.partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y, normals):
    model = model.merge(params)
    preds, grads = jax.vmap(jax.value_and_grad(model))(x)
    loss = jnp.mean(jnp.abs(preds - y) / (jnp.abs(y) + 0.01))
    loss += jnp.mean((grads - normals)**2)
    return loss, model

@jax.jit
def train_step(model, x, y, normals, optimizer):
    params = model.filter(tx.Parameter)
    (loss, model), grads = loss_fn(params, model, x, y, normals)
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
    logging.info(f"[{step}] loss: {loss:.8f}")

def fit(
    model,
    xs,
    ys,
    normals,
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
    for step, (xs_batch, ys_batch, normals_batch) in zip(range(steps), dataloader(xs, ys, normals, batch_size, key=data_key)):
        loss, model, optimizer = train_step(model, xs_batch, ys_batch, normals_batch, optimizer)
        if step % cb_every == 0:
            cb(step, loss, model, optimizer)
        cb(step, loss, model, optimizer)
    return loss, model

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='GDML-JAX MD17 Example')
    parser.add_argument('--data', type=str, default="Dalek.npz")
    parser.add_argument('--model', type=str, default="igr")
    parser.add_argument('--steps', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=128**2)
    parser.add_argument('--cb_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--savegrid', type=int, default=256)
    parser.add_argument('--loglevel', type=int, default=logging.INFO)
    parser.add_argument('--logdir', type=str, default='')
    args = parser.parse_args()

    logdir = args.logdir or f"{Path(args.data).stem}_{args.model}_steps{args.steps}"
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = Path(logdir) / "logs.log"
    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info(args)

    data = load_points(args.data)

    modules = {
        "igr": IGRModel(input_dim=3, depth=7, hidden=512),
        "hash": build_hash_mlp(emb_kwargs={}, hidden=64, act=jax.nn.softplus),
        "idf": IDFModel(
            IGRModel(input_dim=3, depth=7, hidden=512),
            build_hash_mlp(emb_kwargs={}, hidden=64, act=jax.nn.softplus),
            nu=0.04,
        ),
    }
    module = modules[args.model]

    model = module.init(key=jrandom.PRNGKey(0), inputs=data["position"][0])

    def save_callback(step, loss, model, optimizer):
        logging.info(f"[{step}] loss: {loss:.8f}")
        if step % args.save_every == 0:
            modelpath = logdir / f"model{step}.ckpt"
            save_model(modelpath, model)
            vertices, faces = extract_mesh3d(
                model, ngrid=args.savegrid, x_lims=(0,1), y_lims=(0,1), z_lims=(0,1),
            )
            jnp.savez(logdir / f"mesh{step}.npz", vertices=vertices, faces=faces)

    loss, model = fit(
        model,
        xs=data["position"],
        ys=data["distance"],
        normals=data["gradient"],
        steps=args.steps,
        batch_size=args.batch_size,
        cb_every=args.cb_every,
        cb=save_callback,
    )
