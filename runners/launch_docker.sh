#!/bin/bash

DOCKER_IMAGE="evaluating_rewards:latest"
LOCAL_MNT="/mnt/eval_reward"

CMD="bash"
FLAGS=""
if [[ $1 == "jupyter" ]]; then
        FLAGS="-p 127.0.0.1:8888:8888/tcp -v ${LOCAL_MNT}/src:/evaluating-rewards"
        CMD="pip install jupyter && jupyter notebook --ip 0.0.0.0 --allow-root"
fi

docker pull ${DOCKER_IMAGE}
docker run -it --rm \
       -v ${LOCAL_MNT}/data:/root/output \
       -v ${LOCAL_MNT}/mjkey.txt:/root/.mujoco/mjkey.txt \
       ${FLAGS} ${DOCKER_IMAGE} \
       /bin/bash -c "${CMD}"
