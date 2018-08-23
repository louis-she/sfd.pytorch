import sys

import torch

from config import Config
from dataset import create_wf_datasets, my_collate_fn
from model import Net
from trainer import Trainer
from voc_dataset import create_voc_datasets


def main():
    if Config.DATASETS == 'VOC':
        train_dataset, val_dataset = create_voc_datasets(Config.VOC_DATASET_DIR)
    elif Config.DATASETS == 'WF':
        train_dataset, val_dataset = create_wf_datasets(Config.WF_DATASET_DIR)
    else:
        raise RuntimeError('Select a dataset to train in config.py.')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.DATALOADER_WORKER_NUM,
        shuffle=True,
        collate_fn=my_collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=Config.DATALOADER_WORKER_NUM,
        shuffle=False,
        collate_fn=my_collate_fn
    )

    model = Net()
    # optimizer = torch.optim.Adam(
    #     params=model.parameters(),
    #     lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    # )
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY)

    trainer = Trainer(
        optimizer,
        model,
        train_dataloader,
        val_dataloader,
        resume=Config.RESUME_FROM,
        log_dir=Config.LOG_DIR,
        persist_stride=Config.MODEL_SAVE_STRIDE,
        max_epoch=Config.EPOCHS
    )
    trainer.train()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type('torch.FloatTensor')

    main()
