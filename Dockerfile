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

FROM nvidia/cuda:10.0-runtime-ubuntu18.04
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
    rsync \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    fonts-symbola \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update -q \
    && apt-get install -y python3.7-dev python3.7

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN    mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200.zip \
    && unzip mujoco200.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco200.zip

ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/root/.mujoco/mujoco200/bin

WORKDIR /evaluating-rewards
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY ./ci /evaluating-rewards/ci
COPY ./requirements.txt /evaluating-rewards
COPY ./requirements-test.txt /evaluating-rewards

ENV VIRTUAL_ENV=/evaluating-rewards/venv
# mjkey.txt needs to exist for build, but doesn't need to be a real key
RUN touch /root/.mujoco/mjkey.txt && /evaluating-rewards/ci/build_venv.sh $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Delay copying (and installing) the code until the very end
COPY . /evaluating-rewards
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
RUN python setup.py sdist bdist_wheel
RUN pip install dist/evaluating_rewards-*.whl

# Default entrypoints
CMD ["ci/run_tests.sh"]