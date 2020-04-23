"""Metrics based on sampling to approximate a canonical reward in an equivalence class."""

import itertools
import multiprocessing
import multiprocessing.dummy
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import tensorflow as tf

from evaluating_rewards import datasets, rewards, tabular

K = TypeVar("K")


def _make_mesh_tensors(inputs: Mapping[K, np.ndarray]) -> Mapping[K, tf.Tensor]:
    """
    Computes tensors that are the Cartesian product of the inputs.

    The computation takes place in TensorFlow, and so should be more efficient than Python-based
    alternatives like `mesh_to_batch`.

    Args:
        inputs: A mapping from keys to NumPy arrays.

    Returns:
        A mapping from keys to a Tensor, which takes on values from the corresponding
        input NumPy array. Computing the Tensors should yield NumPy arrays equal to
        `{k: itertools.product(inputs.values())[i] for i, k in enumerate(inputs.keys())}`.
    """
    # SOMEDAY(adam): messy, this would be much nicer in TF2 API
    phs = {k: tf.placeholder(v.dtype, shape=v.shape) for k, v in inputs.items()}

    # Increase dimensions for broadcasting
    # So first tensor will be a[:, None, ..., None],
    # second tensor b[None, :, None, ..., None], ...,
    # final tensor z[None, ..., None, :].
    tensors = {}
    for i, (k, ph) in enumerate(phs.items()):
        t = ph
        for j in range(len(phs)):
            if i != j:
                t = tf.expand_dims(t, axis=j)
        tensors[k] = t

    target_shape = tuple((len(v) for v in inputs.values()))
    tensors = {
        k: tf.broadcast_to(t, target_shape + inputs[k].shape[1:]) for k, t in tensors.items()
    }

    target_len = np.product(target_shape)
    tensors = {k: tf.reshape(t, (target_len,) + inputs[k].shape[1:]) for k, t in tensors.items()}
    handles = {k: tf.get_session_handle(t) for k, t in tensors.items()}
    feed_dict = {ph: inputs[k] for k, ph in phs.items()}
    return tf.get_default_session().run(handles, feed_dict=feed_dict)


def mesh_evaluate_models(
    models: Mapping[K, rewards.RewardModel],
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
) -> Mapping[K, np.ndarray]:
    """
    Evaluate models on the Cartesian product of `obs`, `actions`, `next_obs`.

    Computes Cartesian product in TensorFlow for efficiency.

    Args:
        models: A mapping of keys to reward models.
        obs: An array of observations, of shape (l,) + `obs_shape`.
        actions: An array of observations, of shape (m,) + `action_shape`.
        next_obs: An array of observations, of shape (n,) + `obs_shape`.

    Returns:
        A mapping from keys to NumPy arrays of shape (l,m,n), with index (i,j,k) evaluated on the
        transition from `obs[i]` to `next_obs[k]` taking `actions[j]`.
    """
    reward_outputs = {k: m.reward for k, m in models.items()}
    # TODO(adam): this is very memory intensive.
    # Should consider splitting up evaluation into chunks like with `discrete_mean_rews`.
    # Note observations and actions take up much more memory than the evaluated rewards.
    tensors = _make_mesh_tensors(dict(obs=obs, actions=actions, next_obs=next_obs))

    feed_dict = {}
    for m in models.values():
        feed_dict.update({ph: tensors["obs"] for ph in m.obs_ph})
        feed_dict.update({ph: tensors["actions"] for ph in m.act_ph})
        feed_dict.update({ph: tensors["next_obs"] for ph in m.next_obs_ph})

    rews = tf.get_default_session().run(reward_outputs, feed_dict=feed_dict)
    rews = {k: v.reshape(len(obs), len(actions), len(next_obs)) for k, v in rews.items()}
    return rews


def mesh_evaluate_models_slow(
    models: Mapping[K, rewards.RewardModel],
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
) -> Mapping[K, np.ndarray]:
    """
    Evaluate models on the Cartesian product of `obs`, `actions`, `next_obs`.

    Same interface as `mesh_evaluate_models`. This is the unoptimized version, which computes
    the Cartesian product in Python, taking about 20x longer. This is kept around mainly for
    simplicity to aid testing, and for possibility of other optimizations (e.g. JIT like Numba).
    """
    batch = list(itertools.product(obs, actions, next_obs))
    batch = [np.array([m[i] for m in batch]) for i in range(3)]
    batch = rewards.Batch(*batch)
    rews = rewards.evaluate_models(models, batch)
    rews = {k: v.reshape(len(obs), len(actions), len(next_obs)) for k, v in rews.items()}
    return rews


def discrete_iid_evaluate_models(
    models: Mapping[K, rewards.RewardModel],
    obs_dist: datasets.SampleDist,
    act_dist: datasets.SampleDist,
    n_obs: int,
    n_act: int,
) -> Tuple[Mapping[K, np.ndarray], np.ndarray, np.ndarray]:
    """
    Evaluate models on a discretized set of `n_obs` observations and `n_act` actions.

    Args:
        models: A mapping from keys to reward models.
        obs_dist: The distribution to sample observations from.
        act_dist: The distribution to sample actions from.
        n_obs: The number of discrete observations.
        n_act: The number of discrete actions.

    Returns:
        A tuple `(rews, obs, act)` consisting of:
            rews: A mapping from keys to NumPy arrays rew of shape `(n_obs, n_act, n_obs)`, where
                `rew[i, j, k]` is the model evaluated at `obs[i]` to `obs[k]` via action `acts[j]`.
            obs: A NumPy array of shape `(n_obs,) + obs_shape` sampled from `obs_dist`.
            act: A NumPy array of shape `(n_act,) + act_shape` sampled from `act_dist`.
    """
    obs = obs_dist(n_obs)
    act = act_dist(n_act)
    rews = mesh_evaluate_models(models, obs, act, obs)
    return rews, obs, act


def _tile_first_dim(xs: np.ndarray, reps: int) -> np.ndarray:
    reps_d = (reps,) + (1,) * (xs.ndim - 1)
    return np.tile(xs, reps_d)


def sample_mean_rews(
    models: Mapping[K, rewards.RewardModel],
    mean_from_obs: np.ndarray,
    act_samples: np.ndarray,
    next_obs_samples: np.ndarray,
    batch_size: int = 2 ** 28,
) -> Mapping[K, np.ndarray]:
    """
    Estimates the mean reward from observations `mean_from_obs` using given samples.

    Evaluates in batches of at most `batch_size` bytes to avoid running out of memory. Note that
    the observations and actions, being vectors, often take up much more memory in RAM than the
    results, a scalar value.

    Args:
        models: A mapping from keys to reward models.
        mean_from_obs: Observations to compute the mean starting from.
        act_samples: Actions to compute the mean with respect to.
        next_obs_samples: Next observations to compute the mean with respect to.
        batch_size: The maximum number of points to compute the reward with respect to in a single
            batch.

    Returns:
        A mapping from keys to NumPy array of shape `(len(mean_from_obs),)`, containing the
        mean reward of the model over triples:
            `(obs, act, next_obs) for act, next_obs in zip(act_samples, next_obs_samples)`
    """
    assert act_samples.shape[0] == next_obs_samples.shape[0]
    assert mean_from_obs.shape[1:] == next_obs_samples.shape[1:]

    # Compute indexes to not exceed batch size
    sample_mem_usage = act_samples.nbytes + mean_from_obs.nbytes
    obs_per_batch = batch_size // sample_mem_usage
    if obs_per_batch <= 0:
        msg = f"`batch_size` too small to compute a batch: {batch_size} < {sample_mem_usage}."
        raise ValueError(msg)
    idxs = np.arange(0, len(mean_from_obs), obs_per_batch)
    idxs = np.concatenate((idxs, [len(mean_from_obs)]))  # include end point

    # Compute mean rewards
    mean_rews = {k: [] for k in models.keys()}
    reps = min(obs_per_batch, len(mean_from_obs))
    act_tiled = _tile_first_dim(act_samples, reps)
    next_obs_tiled = _tile_first_dim(next_obs_samples, reps)
    for start, end in zip(idxs[:-1], idxs[1:]):
        obs = mean_from_obs[start:end]
        obs_repeated = np.repeat(obs, len(act_samples), axis=0)
        batch = rewards.Batch(
            obs=obs_repeated,
            actions=act_tiled[: len(obs_repeated), :],
            next_obs=next_obs_tiled[: len(obs_repeated), :],
        )
        rews = rewards.evaluate_models(models, batch)
        rews = {k: v.reshape(len(obs), -1) for k, v in rews.items()}
        for k, m in mean_rews.items():
            means = np.mean(rews[k], axis=1)
            m.extend(means)

    mean_rews = {k: np.array(v) for k, v in mean_rews.items()}
    for v in mean_rews.values():
        assert v.shape == (len(mean_from_obs),)
    return mean_rews


def sample_canon_shaping(
    models: Mapping[K, rewards.RewardModel],
    batch: rewards.Batch,
    act_dist: datasets.SampleDist,
    obs_dist: datasets.SampleDist,
    n_mean_samples: int,
    discount: float = 1.0,
) -> Mapping[K, np.ndarray]:
    r"""
    Canonicalize `batch` for `models` using a sample-based estimate of mean reward.

    Specifically, the algorithm works by sampling `n_mean_samples` from `act_dist` and `obs_dist`
    to form a dataset of pairs $D = \{(a,s')\}$. We then consider a transition dynamics where,
    for any state $s$, the probability of transitioning to $s'$ after taking action $a$ is given by
    its measure in $D$. The policy takes actions $a$ independent of the state given by the measure
    of $(a,\cdot)$ in $D$.

    This gives value function:
        \[V(s) = \expectation_{(a,s') \sim D}\left[R(s,a,s') + \gamma V(s')\right]\].
    The resulting shaping works out to be:
        \[F(s,a,s') = \gamma \expectation_{(a',s'') \sim D}\left[R(s',a',s'')\right]
                    - \expectation_{(a,s') \sim D}\left[R(s,a,s')\right]
                    - \gamma \expectation_{(s, \cdot) \sim D, (a,s') \sim D}\left[R(s,a,s')\right]
        \].

    If `batch` was a mesh of $S \times A \times S$ and $D$ is a mesh on $A \times S$,
    where $S$ and $A$ are i.i.d. sampled from some observation and action distributions, then this
    is the same as discretizing the reward model by $S$ and $A$ and then using
    `tabular.fully_connected_random_canonical_reward`. The action and next-observation in $D$ are
    sampled i.i.d., but since we are not computing an entire mesh, the sampling process introduces a
    faux dependency. Additionally, `batch` may have an arbitrary distribution.

    Empirically, however, the two methods produce very similar results. The main advantage of this
    method is its computational efficiency, for similar reasons to why random search is often
    preferred over grid search when some unknown subset of parameters are relatively unimportant.

    Args:
        models: A mapping from keys to reward models.
        batch: A batch to evaluate the models with respect to.
        act_dist: The distribution to sample actions from.
        obs_dist: The distribution to sample next observations from.
        n_mean_samples: The number of samples to take.
        discount: The discount parameter to use for potential shaping.

    Returns:
        A mapping from keys to NumPy arrays containing rewards from the model evaluated on batch
        and then canonicalized to be invariant to potential shaping and scale.
    """
    raw_rew = rewards.evaluate_models(models, batch)

    # Sample-based estimate of mean reward
    act_samples = act_dist(n_mean_samples)
    next_obs_samples = obs_dist(n_mean_samples)

    all_obs = np.concatenate((next_obs_samples, batch.obs, batch.next_obs), axis=0)
    unique_obs, unique_inv = np.unique(all_obs, return_inverse=True, axis=0)
    mean_rews = sample_mean_rews(models, unique_obs, act_samples, next_obs_samples)
    mean_rews = {k: v[unique_inv] for k, v in mean_rews.items()}

    dataset_mean_rews = {k: v[0:n_mean_samples] for k, v in mean_rews.items()}
    total_mean = {k: np.mean(v) for k, v in dataset_mean_rews.items()}

    batch_mean_rews = {k: v[n_mean_samples:].reshape(2, -1) for k, v in mean_rews.items()}

    # Use mean rewards to canonicalize reward up to shaping
    deshaped_rew = {}
    for k in models.keys():
        raw = raw_rew[k]
        mean = batch_mean_rews[k]
        total = total_mean[k]
        mean_obs = mean[0, :]
        mean_next_obs = mean[1, :]
        # Note this is the only part of the computation that depends on discount, so it'd be
        # cheap to evaluate for many values of `discount` if needed.
        deshaped = raw + discount * mean_next_obs - mean_obs - discount * total
        deshaped *= tabular.canonical_scale(deshaped)
        deshaped_rew[k] = deshaped

    return deshaped_rew


def cross_distance(
    rewxs: Mapping[Any, np.ndarray],
    rewys: Mapping[Any, np.ndarray],
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    parallelism: Optional[int] = None,
    threading: bool = True,
) -> pd.DataFrame:
    """Helper function to compute distance between all pairs of rewards from `rewxs` and `rewys`.

    Args:
        rewxs: A mapping from keys to NumPy arrays of shape `(n,)`.
        rewys: A mapping from keys to NumPy arrays of shape `(n,)`.
        distance_fn: A function to compute the distance between two NumPy arrays.
        parallelism: The number of threads/processes to execute in parallel; if not specified,
            defaults to `multiprocessing.cpu_count()`.
        threading: If true, use multi-threading; otherwise, use multiprocessing. For many NumPy
            functions, multi-threading is higher performance since NumPy releases the GIL, and
            threading avoids expensive copying of the arrays needed for multiprocessing.

    Returns:
        A square DataFrame whose columns and indices consist of `rews.keys()`, with each cell
        `(i,j)` consisting of `distance_fn(rews[i], rews[j])`.
    """
    shapes = set((v.shape for v in rewxs.values()))
    shapes.update((v.shape for v in rewys.values()))
    assert len(shapes) <= 1, "rewards differ in shape"

    tasks = {(kx, ky): (rewx, rewy) for kx, rewx in rewxs.items() for ky, rewy in rewxs.items()}

    if parallelism == 1:
        # Only one process? Skip multiprocessing, since creating Pool adds overhead.
        results = [distance_fn(rewx, rewy) for rewx, rewy in tasks.values()]
    else:
        # We want parallelism, use multiprocessing to speed things up.
        module = multiprocessing.dummy if threading else multiprocessing
        with module.Pool(processes=parallelism) as pool:
            results = pool.starmap(distance_fn, tasks.values())

    results = dict(zip(tasks.keys(), results))
    return pd.Series(results).unstack()
