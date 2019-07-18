#! /bin/bash
IMAGE_NAME="tensor-flow"
CONTAINER_NAME="tensor-flow-gpu"
sudo /usr/bin/nvidia-docker run --name ${CONTAINER_NAME} -tid  ${IMAGE_NAME} /sbin/init
