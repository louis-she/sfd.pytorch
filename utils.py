import numpy as np

def change_coordinate(coordinates):
    """change top left bottom right to center x center y, width, height"""
    width = (coordinates[:, 3] - coordinates[:, 1])[:, np.newaxis]
    height = (coordinates[:, 2] - coordinates[:, 0])[:, np.newaxis]
    center_x = ((coordinates[:, 3] + coordinates[:, 1]) / 2)[:, np.newaxis]
    center_y = ((coordinates[:, 2] + coordinates[:, 0]) / 2)[:, np.newaxis]
    return np.concatenate([center_x, center_y, width, height], axis=1)