import argparse
import os
import sys

import cv2
import numpy as np
import torch

from anchor import generate_anchors
from config import Config
from model import Net
from utils import change_coordinate, change_coordinate_inv, seek_model, save_bounding_boxes_image

device = torch.device(Config.DEVICE)


class Detector(object):

    def __init__(self, model, image_size=Config.IMAGE_SIZE, keep=200):
        checkpoint = torch.load(seek_model(model))
        self.model = Net().to(device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.keep = keep
        self.image_size = image_size

    def infer(self, image):
        image = cv2.imread(image)
        scale = (image.shape[0] / self.image_size,
                 image.shape[1] / self.image_size)

        image = cv2.resize(image, (self.image_size,) * 2)
        _input = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

        predictions = self.model(_input)
        # flatten predictions
        for index, prediction in enumerate(predictions):
            predictions[index] = prediction.view(6, -1).permute(1, 0)
        predictions = torch.cat(predictions)

        # get sorted indices by score
        diff = predictions[:, 5] - predictions[:, 4]
        _, indices = torch.sort(diff, descending=True)

        # sort and slice predictions
        predictions = predictions[indices][:self.keep]

        # generate anchors then sort and slice
        anchor_configs = (
            Config.ANCHOR_STRIDE,
            Config.ANCHOR_SIZE,
            Config.IMAGE_SIZE
        )
        anchors = change_coordinate(np.vstack(
            list(map(lambda x: np.array(x), generate_anchors(*anchor_configs)))
        ))
        anchors = torch.tensor(anchors[indices][:self.keep]).float().to(device)

        x = (predictions[:, 0] * anchors[:, 2] + anchors[:, 0]) * scale[1]
        y = (predictions[:, 1] * anchors[:, 3] + anchors[:, 1]) * scale[0]
        w = (torch.exp(predictions[:, 2]) * anchors[:, 2]) * scale[1]
        h = (torch.exp(predictions[:, 3]) * anchors[:, 3]) * scale[0]

        bounding_boxes = torch.stack((x, y, w, h), dim=1).cpu().data.numpy()
        bounding_boxes = change_coordinate_inv(bounding_boxes)

        # TODO: do non-maximum suppression for bounding_boxes here

        return bounding_boxes


def main(args):
    print('predicted bounding boxes of faces:')
    bboxes = Detector(args.model).infer(args.image)
    print(bboxes)
    if args.save_to:
        save_bounding_boxes_image(args.image, bboxes, args.save_to)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--image', type=str,
                        help='image to be predicted')
    parser.add_argument('--model', type=str,
                        help='model to use, could be epoch number, model file '
                             'name or model file absolute path')
    parser.add_argument('--keep', type=int, default=150,
                        help='how many predictions to keep, default: 150')
    parser.add_argument('--save_to', type=str,
                        help='save the image with bboxes to a file')

    args = parser.parse_args()
    main(args)
