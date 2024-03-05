#!/bin/bash

#!/bin/bash

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)

docker build $PROJECT_ROOT_DIR -f $PROJECT_ROOT_DIR/docker/Dockerfile \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t nano:latest \
    --progress plain 
