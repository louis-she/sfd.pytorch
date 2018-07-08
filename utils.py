import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from config import Config


def change_coordinate(coordinates):
    """change top left bottom right to center x center y, width, height"""
    width = (coordinates[:, 3] - coordinates[:, 1])[:, np.newaxis]
    height = (coordinates[:, 2] - coordinates[:, 0])[:, np.newaxis]
    center_x = ((coordinates[:, 3] + coordinates[:, 1]) / 2)[:, np.newaxis]
    center_y = ((coordinates[:, 2] + coordinates[:, 0]) / 2)[:, np.newaxis]
    return np.concatenate([center_x, center_y, width, height], axis=1)


def change_coordinate_inv(coordinates):
    """center_x, center_y, width, height to top, left, bottom, right"""
    top = (coordinates[:, 1] - coordinates[:, 3] / 2)[:, np.newaxis]
    left = (coordinates[:, 0] - coordinates[:, 2] / 2)[:, np.newaxis]
    bottom = (coordinates[:, 1] + coordinates[:, 3] / 2)[:, np.newaxis]
    right = (coordinates[:, 0] + coordinates[:, 3] / 2)[:, np.newaxis]
    return np.concatenate([top, left, bottom, right], axis=1)


def seek_model(file_name):
    log_dir = Config.LOG_DIR

    candidate_a = os.path.join(log_dir, 'models', file_name)
    candidate_b = os.path.join(log_dir, 'models', 'epoch_{}.pth.tar'.format(file_name))
    candidate_c = file_name

    if os.path.isfile(candidate_a):
        state_file = candidate_a
    elif os.path.isfile(candidate_b):
        state_file = candidate_b
    elif os.path.isfile(candidate_c):
        state_file = candidate_c
    else:
        raise RuntimeError(
            "model file {} is not found".format(resume)
        )

    return state_file


def draw_bounding_boxes(image_path, bounding_boxes):
    """draw bounding box on a image, should only be called in
    jupyter notebook context
    """
    image = cv2.resize(cv2.imread(image_path), (Config.IMAGE_SIZE,) * 2)[:, :, ::-1]
    fig, ax = plt.subplots(1)
    for bbox in bounding_boxes:
        rect = patches.Rectangle(
            (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.imshow(image)
