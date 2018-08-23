import argparse
import os
import sys

import cv2
import numpy as np
import torch

from anchor import generate_anchors, anchors_of_feature_map
from config import Config
from model import Net
from utils import change_coordinate, change_coordinate_inv, seek_model, save_bounding_boxes_image, nms
from evaluation_metrics import softmax

device = torch.device(Config.DEVICE)


class Detector(object):

    def __init__(self, model, image_size=Config.IMAGE_SIZE, threshold=Config.PREDICTION_THRESHOLD):
        if type(model) == str:
            checkpoint = torch.load(seek_model(model))
            self.model = Net().to(device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            self.model = model
        self.model.eval()
        self.threshold = threshold
        self.image_size = image_size

        anchor_configs = (
            Config.ANCHOR_STRIDE,
            Config.ANCHOR_SIZE,
        )

    def convert_predictions(self, predictions, path, anchors):
        # get sorted indices by score

        scores, klass = torch.max(softmax(predictions[:, 4:]), dim=1)
        inds = klass != 0

        scores, klass, predictions, anchors = \
            scores[inds], klass[inds], predictions[inds], anchors[inds]

        if len(scores) == 0:
            return None

        scores, inds = torch.sort(scores, descending=True)
        klass, predictions, anchors = klass[inds], predictions[inds], anchors[inds]

        # inds = scores > self.threshold
        # scores, klass, predictions, anchors = \
        #     scores[inds], klass[inds], predictions[inds], anchors[inds]

        scores, klass, predictions, anchors = \
            scores[:200], klass[:200], predictions[:200], anchors[:200]

        if len(predictions) == 0:
            return None
        anchors = anchors.to(device).float()

        x = (predictions[:, 0] * anchors[:, 2] + anchors[:, 0])
        y = (predictions[:, 1] * anchors[:, 3] + anchors[:, 1])
        w = (torch.exp(predictions[:, 2]) * anchors[:, 2])
        h = (torch.exp(predictions[:, 3]) * anchors[:, 3])

        bounding_boxes = torch.stack((x, y, w, h), dim=1).cpu().data.numpy()
        bounding_boxes = change_coordinate_inv(bounding_boxes)

        scores = scores.cpu().data.numpy()
        klass = klass.cpu().data.numpy()
        bboxes_scores = np.hstack(
            (bounding_boxes, np.array(list(zip(*(scores, klass)))))
        )

        # nms
        keep = nms(bboxes_scores)
        return bboxes_scores[keep]

    def forward(self, batched_data):
        """predict with pytorch dataset output

        Args:
            batched_data (tensor): yield by the dataset
        Returns: predicted coordinate and score
        """
        images = batched_data[0].permute(0, 3, 1, 2).to(device).float()
        predictions = list(zip(*list(self.model(images))))
        result = []

        for i, prediction in enumerate(predictions):
            prediction = list(prediction)
            anchors = []
            for k, feature_map_prediction in enumerate(prediction):
                # create anchors of this feature_map_prediction layer

                if (k % 2) == 0:
                    anchors.append( np.array( anchors_of_feature_map(
                        Config.ANCHOR_STRIDE[k//2],
                        Config.ANCHOR_SIZE[k//2],
                        feature_map_prediction.size()[1:])))

                prediction[k] = feature_map_prediction \
                    .view(feature_map_prediction.size()[0], -1) \
                    .permute(1, 0).contiguous()

            reg_preds = torch.cat(prediction[::2])
            cls_preds = torch.cat(prediction[1::2])

            anchors = torch.tensor(np.vstack(anchors))

            result.append(self.convert_predictions(
                torch.cat((reg_preds, cls_preds), dim=1),
                batched_data[2][i], anchors))

        return result

    def infer(self, image):
        image = cv2.imread(image)
        image = image - np.array([104, 117, 123], dtype=np.uint8)

        _input = torch.tensor(image).permute(2, 0, 1).float() \
            .to(device).unsqueeze(0)

        predictions = self.model(_input)
        # flatten predictions
        reg_preds = []
        cls_preds = []
        anchors = []
        for index, prediction in enumerate(predictions):
            if (index % 2) == 0:
                anchors.append( np.array( anchors_of_feature_map (
                    Config.ANCHOR_STRIDE[index//2],
                    Config.ANCHOR_SIZE[index//2],
                    prediction.size()[2:]
                )))

            predictions[index] = prediction.squeeze().view(prediction.size()[1], -1).permute(1, 0)

        anchors = torch.tensor(np.vstack(anchors))
        reg_preds = torch.cat(predictions[::2])
        cls_preds = torch.cat(predictions[1::2])

        return self.convert_predictions(torch.cat((reg_preds, cls_preds), dim=1), None, anchors)


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
