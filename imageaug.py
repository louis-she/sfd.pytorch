import numpy as np
from random import random

def crop_square(image, coordinates, ratio=1, keep_area_threshold=0.5):
    """random crop a image into a square image and change the
    original coordinates to new coordinates. Some coordinates will be last
    if it is at outside of the cropped area.

    Args:
        image (ndarray): numpy image, should be [height, width, channel]
        coordinates (tuple): a tuple of coordinates list, should be
            ([top, left, bottom, right], ...)
        ratio (int, optional): defaults to 1. cropped ratio, relative to the
            shorter edge of the image
        keep_area_threshold (float, optional): defaults to 0.5. how much area
            in the cropped size of a ground truth bounding box to decide whther
            to keep it.
    Returns:
        tuple: (cropped_image, new_coordinates), noticed that new_coordinates
            may be an empty list.
    """

    size = image.shape[:2]
    short_size = np.min(size)
    square_size = int(short_size * ratio)

    n_top = int((image.shape[0] - square_size) * random())
    n_left = int((image.shape[1] - square_size) * random())
    n_bottom = n_top + square_size
    n_right = n_left + square_size

    cropped_image = image[n_top:n_bottom, n_left:n_right]

    new_coordinates = []
    for coordinate in coordinates:
        width = coordinate[3] - coordinate[1]
        height = coordinate[2] - coordinate[0]
        n_width = max(min(coordinate[3], n_right) - max(coordinate[1], n_left), 0)
        n_height = max(min(coordinate[2], n_bottom) - max(coordinate[0], n_top), 0)

        # there are some all zero coordinates in wider face
        if (width * height) == 0:
            continue
        area_in_crop = (n_width * n_height) / (width * height)

        if area_in_crop < keep_area_threshold:
           continue

        new_coordinates.append([
            max(coordinate[0] - n_top, 0),
            max(coordinate[1] - n_left, 0),
            max(coordinate[2] - n_top, 0),
            max(coordinate[3] - n_left, 0),
            *coordinate[4:]
        ])

    return cropped_image, new_coordinates

def random_horizontal_flip(image, coordinates):
    """randomly horizontal flip a image and its coodinates

    Args:
        image (ndarray): numpy image, should be [height, width, channel]
        coordinates (tuple): a tuple of coordinates list, should be
            ([top, left, bottom, right], ...)
    Returns:
        tuple: (image, new_coordinates), noticed that new_coordinates
            may be an empty list.
    """

    if random() > 0.5:
        return image, coordinates
    image = image[:, ::-1, :]
    new_coordinates = []
    for coordinate in coordinates:
        new_coordinates.append([
            coordinate[0],
            image.shape[1] - coordinate[1],
            coordinate[2],
            image.shape[1] - coordinate[3],
            *coordinate[4:]
        ])
    return image, new_coordinates