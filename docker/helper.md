sudo nvidia-docker run -it --shm-size 1G \
-v /PATH_TO_DATASETS/datasets:/datasets \
-v /PATH_TO_LOG/log:/log \
-v /PATH_TO_VGG16_PRETRAINED_WEIGHTS/.torch:/root/.torch \
image_id(eg.:0a4e7e23a9b6) python ./sfd.pytorch/main.py