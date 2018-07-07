import numpy as np

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