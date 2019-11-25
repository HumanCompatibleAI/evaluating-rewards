#!/bin/bash
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

DOCKER_IMAGE="humancompatibleai/evaluating_rewards:latest"
LOCAL_MNT="/mnt/eval_reward"

CMD="bash"
FLAGS=""
if [[ $1 == "jupyter" ]]; then
        FLAGS="-p 127.0.0.1:8888:8888/tcp"
        CMD="pip install jupyter && jupyter notebook --ip 0.0.0.0 --allow-root"
fi

docker pull ${DOCKER_IMAGE}
docker run -it --rm \
       -v ${LOCAL_MNT}/data:/root/output \
       -v ${LOCAL_MNT}/src:/evaluating-rewards \
       -v ${LOCAL_MNT}/mjkey.txt:/root/.mujoco/mjkey.txt \
       ${FLAGS} ${DOCKER_IMAGE} \
       /bin/bash -c "${CMD}"
