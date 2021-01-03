#!/bin/bash

DATA_ROOT=data
COPY_DIRS="train_experts/ground_truth/20201203_105631_297835/ train_experts/point_maze_wrong_target/20201122_053216_fb1b0e/"
S3_REPO="s3://evaluating-rewards"

for dir in ${COPY_DIRS}; do
    aws s3 sync ${DATA_ROOT}/${dir} ${S3_REPO}/${dir}
done
aws s3 sync --exclude rl ${DATA_ROOT}/transfer_point_maze/ ${S3_REPO}/transfer_point_maze/
