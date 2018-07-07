import torch
import torch.backends.cudnn as cudnn
import cv2
from dataset import create_datasets, my_collate_fn
from model import Net
from trainer import Trainer

cudnn.benchmark = True
torch.set_default_tensor_type('torch.DoubleTensor')

if __name__ == "__main__":

    train_dataset, val_dataset = create_datasets('/home/louis/datasets/wider_face')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        collate_fn=my_collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False,
        collate_fn=my_collate_fn
    )

    model = Net()
    trainables_wo_bn = [param for name, param in model.named_parameters() if
                        param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if
                          param.requires_grad and 'bn' in name]

    optimizer = torch.optim.Adam([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=0.001)

    trainer = Trainer(
        optimizer,
        model,
        train_dataloader,
        val_dataloader,
        max_epoch=100,
        resume=1,
        log_dir='/media/louis/ext4/models/sfd'
    )
    trainer.train()
