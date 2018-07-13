import torch
from config import Config
from model import Net
from dataset import create_wf_datasets, my_collate_fn
from utils import change_coordinate, seek_model
from detector import Detector
import argparse


def main(args):
    _, val_dataset = create_wf_datasets(Config.WF_DATASET_DIR)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.DATALOADER_WORKER_NUM,
        shuffle=False,
        collate_fn=my_collate_fn
    )

    model = Net()
    state_file = seek_model(args.model)
    checkpoint = torch.load(state_file)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    for image, gt, _ in val_dataloader:
        print(_)
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--model', type=str,
                        help='model to use, could be epoch number, model file '
                             'name or model file absolute path')

    args = parser.parse_args()
    main(args)
