XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth #/tmp/.Xauthority

# Run Docker container
docker run -d -it \
    --gpus all \
    --privileged \
    --network=host \
    --shm-size=15G \
    --device=/dev \
    --runtime=nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTH \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    -v ./..:/home/GSViewer \
    --name gsviewer_exp1 \
    gsviewer_image