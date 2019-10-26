# Copyright 2019 DeepMind Technologies Limited
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

"""Install script for setuptools."""

import setuptools
import src.evaluating_rewards


def load_requirements(fname):
    with open(fname) as f:
        return f.read().strip().split("\n")


setuptools.setup(
    name="evaluating_rewards",
    version=src.evaluating_rewards.__version__,
    description="Evaluating and comparing reward models.",
    author="DeepMind Technologies Limited and Adam Gleave",
    author_email="adam@gleave.me",
    python_requires=">=3.6.0",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    package_data={"evaluating_rewards": ["py.typed"]},
    install_requires=load_requirements("requirements.txt"),
    extras_require={"test": load_requirements("requirements-dev.txt")},
    url="https://github.com/AdamGleave/evaluating_rewards",
    license="Apache License, Version 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
