# Copyright 2019 Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom policies and policy wrappers."""

import contextlib
from typing import Iterable, Iterator

from imitation.policies import serialize
import numpy as np
from stable_baselines.common import policies, vec_env


def _validate_policies(pola: policies.BasePolicy, polb: policies.BasePolicy, idx: int) -> None:
    """Checks policies have consistent fields."""
    fields = ["ob_space", "ac_space", "n_env", "n_batch", "n_steps"]
    wildcard_fields = ["n_env", "n_batch", "n_steps"]
    for f in fields:
        pola_val = getattr(pola, f)
        polb_val = getattr(polb, f)
        different = pola_val != polb_val
        either_none = pola_val is None or polb_val is None  # None indicates indifference
        is_wildcard = f in wildcard_fields
        if different and not (either_none and is_wildcard):
            raise ValueError(
                f"Inconsistent value for '{f}': '{pola_val}' at 0" f" != '{polb_val}' at {idx}"
            )


class PolicyMixture(policies.BasePolicy):
    """Creates a stochastic policy from a mixture of policies.

    WARNING: The resulting policy is only suitable for inference and not training.
    """

    def __init__(
        self, pols: Iterable[policies.BasePolicy], transition_p: float = 0.1, seed: int = 0
    ):  # pylint:disable=super-init-not-called
        """Constructs a PolicyMixture.

        Arguments:
            pols: Policies to sample from.
            transition_p: The probability of, at any given time step, switching from one
                policy to another.
            seed: Seed for the internal random generator, choosing when to switch between policies
                and which policy to switch to.

        Raises:
              ValueError if policies is empty, or the policies have inconsistent attributes.
              ValueError if transition_p is not a probability.
        """
        self.policies = list(pols)
        if not self.policies:
            raise ValueError("policies must be non-empty.")
        a_policy = self.policies[0]
        for idx, pol in enumerate(self.policies[1:]):
            _validate_policies(a_policy, pol, idx)

        # Set attributes expected of a policy. Do not call super() since that creates placeholders
        # which we do not want (and are only needed for training.)
        self.sess = None
        self.ob_space = a_policy.ob_space
        self.ac_space = a_policy.ac_space
        self.n_env = a_policy.n_env
        self.n_steps = a_policy.n_steps
        self.n_batch = a_policy.n_batch

        # Randomly choose between policies
        self.transition_p = transition_p
        if not 0 <= self.transition_p <= 1:
            raise ValueError(f"Probability transition_p = {transition_p} must be in [0,1].")
        self.rng = np.random.RandomState(seed)
        self.current_pol = self.rng.choice(self.policies)

    def _maybe_change_pol(self):
        if self.rng.uniform() < self.transition_p:
            self.current_pol = self.rng.choice(self.policies)

    def step(self, obs, state=None, mask=None, **kwargs):
        self._maybe_change_pol()
        return self.current_pol.step(obs, state=state, mask=mask, **kwargs)

    def proba_step(self, obs, state=None, mask=None, **kwargs):
        self._maybe_change_pol()
        return self.current_pol.proba_step(obs, state=state, mask=mask, **kwargs)


@contextlib.contextmanager
def load_mixture(policy_path: str, venv: vec_env.VecEnv) -> Iterator[policies.BasePolicy]:
    """Load a mixed policy.

    Arguments:
        policy_path: A ':'-separated string, the first argument of which is `transition_p`, the
            probability of switching policy at each timestep; the remaining arguments are pairs of
            policy_type and policy_type that are used to load additional policies.
        venv: An environment that the policy is to be used with.
    """
    transition_p, *pol_args = policy_path.split(":")
    transition_p = float(transition_p)
    pol_types = pol_args[::2]
    pol_paths = pol_args[1::2]
    if len(pol_types) != len(pol_paths):
        raise ValueError(f"Malformed policy_path, missing policy path after type: '{policy_path}'")

    with contextlib.ExitStack() as stack:
        pols = [
            stack.enter_context(serialize.load_policy(inner_type, inner_path, venv))
            for inner_type, inner_path in zip(pol_types, pol_paths)
        ]
        mixture = PolicyMixture(pols, transition_p=transition_p)
        yield mixture
