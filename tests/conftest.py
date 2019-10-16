# Copyright 2019 DeepMind Technologies Limited and Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration settings and fixtures for tests."""

import pytest
import tensorflow as tf
from tests import common


@pytest.fixture(name="graph")
def fixture_graph():
    graph = tf.Graph()
    yield graph


@pytest.fixture(name="session")
def fixture_session(graph):
    with tf.Session(graph=graph) as session:
        yield session


env = pytest.fixture(common.make_env)
venv = pytest.fixture(common.make_venv)
