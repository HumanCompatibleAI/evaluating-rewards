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

# base stage contains just binary dependencies.
# This is used in the CI build.
FROM nvidia/cuda:10.0-runtime-ubuntu18.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

RUN    apt-get update -q \
    && apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    ffmpeg \
    software-properties-common \
    net-tools \
    parallel \
    python3.7 \
    python3.7-dev \
    rsync \
    unzip \
    vim \
    virtualenv \
    xpra \
    xserver-xorg-dev \
    fonts-symbola \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN    mkdir -p /root/.mujoco \
    && curl -o mjpro150.zip https://www.roboti.us/download/mjpro150_linux.zip \
    && unzip mjpro150.zip -d /root/.mujoco \
    && rm mjpro150.zip

# Set the PATH to the venv before we create the venv, so it's visible in base.
# This is since we may create the venv outside of Docker, e.g. in CI
# or by binding it in for local development.
ENV PATH="/venv/bin:$PATH"
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}

# python-req stage contains Python venv, but not code.
# It is useful for development purposes: you can mount
# code from outside the Docker container.
FROM base as python-req

WORKDIR /evaluating-rewards
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY ./scripts /evaluating-rewards/scripts
COPY ./requirements.txt /evaluating-rewards
COPY ./requirements-dev.txt /evaluating-rewards

# mjkey.txt needs to exist for build, but doesn't need to be a real key
RUN touch /root/.mujoco/mjkey.txt && /evaluating-rewards/scripts/build_venv.sh /venv

# full stage contains everything.
# Can be used for deployment and local testing.
FROM python-req as full

# Delay copying (and installing) the code until the very end
COPY . /evaluating-rewards
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
RUN python setup.py sdist bdist_wheel
RUN pip install dist/evaluating_rewards-*.whl

# Default entrypoints
CMD ["pytest", "-n", "auto", "-vv", "tests/", "examples/"]
