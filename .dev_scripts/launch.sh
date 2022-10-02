CODE_DIR=/home/ayushman/pogchamps_corn/code
RUN_DIR=/mnt/hdd1/pogchamps_corn/
DATA_DIR=/home/ayushman/pogchamps_corn/input/
CONTAINER=pogchamps_corn

docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8888:8888 --gpus all --ipc=host \
-v $CODE_DIR:/workspace \
-v $DATA_DIR:/data/ \
-v $RUN_DIR:/runs/ $CONTAINER