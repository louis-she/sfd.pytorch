import torch
from config import Config
from model import Net
from dataset import create_wf_datasets, my_collate_fn
from utils import change_coordinate, seek_model
from detector import Detector
import argparse
from evaluation_metrics import AP
import numpy as np


def main(args):
    _, val_dataset = create_wf_datasets(Config.WF_DATASET_DIR)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.DATALOADER_WORKER_NUM,
        shuffle=False,
        collate_fn=my_collate_fn
    )

    total = len(val_dataloader)
    detector = Detector(args.model)
    APs = []
    for index, data in enumerate(val_dataloader):
        predictions = detector.forward(data)
        for i in range(len(predictions)):
            if predictions[i] is None:
                APs.append(0)
                continue
            prediction = predictions[i]
            gt = data[1][i]
            scale = data[3][i]

            gt[:, 0] *= scale[0]
            gt[:, 1] *= scale[1]
            gt[:, 2] *= scale[0]
            gt[:, 3] *= scale[1]

            ap = AP(prediction, gt, 0.5)
            APs.append(ap[1.0])

        print("{} / {}".format(index, total))

    print("mAP: {}".format(sum(APs)/len(APs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--model', type=str,
                        help='model to use, could be epoch number, model file '
                             'name or model file absolute path')

    args = parser.parse_args()
    main(args)
