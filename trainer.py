import os

import torch
import torch.nn.functional as F
import numpy as np

from utils import change_coordinate
from anchor import generate_anchors, mark_anchors

device = torch.device("cuda")


class Trainer(object):

    def __init__(self, optimizer, model, training_dataloader,
                 validation_dataloader, log_dir=False, max_epoch=100,
                 resume=False, persist_stride=5):

        self.log_dir = log_dir
        self.optimizer = optimizer
        self.model = model.to(device)
        self.max_epoch = max_epoch
        self.resume = resume
        self.persist_stride = persist_stride
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.start_epoch = 1
        self.current_epoch = 1
        self.anchors = np.vstack(
            list(map(lambda x: np.array(x), generate_anchors()))
        )

        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if resume:
            state_file = os.path.join(self.log_dir, 'models', resume)
            if not os.path.isfile(state_file):
                raise RuntimeError(
                    "resume file {} is not found".format(state_file))
            print("loading checkpoint {}".format(state_file))
            checkpoint = torch.load(state_file)
            self.start_epoch = self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint {} (epoch {})".format(
                state_file, self.current_epoch))

    def train(self):
        for self.current_epoch in range(self.start_epoch, self.max_epoch+1):
            self.run_epoch(mode='train')
            if self.validation_dataloader:
                self.run_epoch(mode='validate')
            if not (self.current_epoch % self.persist_stride):
                self.persist()

    def run_epoch(self, mode):
        if mode == 'train':
            dataloader = self.training_dataloader
            self.model.train()
        else:
            dataloader = self.validation_dataloader
            self.model.eval()

        with torch.set_grad_enabled(mode == 'train'):
            for images, gt_bboxes, file_path in dataloader:
                # gt_bboxes: 2-d list of (batch_size, ndarray(bbox_size, 4) )
                image = images.permute(0, 3, 1, 2).double().to(device)

                predictions = list(self.model(image))
                for index, prediction in enumerate(predictions):
                    predictions[index] = torch.squeeze(prediction).view(6, -1).permute(1, 0)
                predictions = torch.cat(predictions)
                gt_bboxes = gt_bboxes[0]

                # pick up positive and negative anchors
                pos_indices, gt_bboxes_indices, neg_indices = \
                    mark_anchors(self.anchors, gt_bboxes)

                # make samples of negative to positive 3:1
                n_neg_indices = len(pos_indices) * 3
                reg_random_indices = torch.randperm(len(neg_indices))
                neg_indices = neg_indices[reg_random_indices][:n_neg_indices]

                pos_anchors = torch.tensor(
                    change_coordinate(self.anchors[pos_indices])
                ).double().to(device) / 640

                neg_anchors = torch.tensor(
                    change_coordinate(self.anchors[neg_indices])
                ).double().to(device) / 640

                pos_preds = predictions[pos_indices].to(device)
                neg_preds = predictions[neg_indices].to(device)

                epsilon = 0.0001
                # equation 5 in the paper faster rcnn
                print(pos_preds.size())
                print(pos_anchors.size())
                print(file_path)
                tx = pos_preds[:, 0] - pos_anchors[:, 0]
                ty = pos_preds[:, 1] - pos_anchors[:, 1]
                tw = torch.log((pos_preds[:, 2] + epsilon) / pos_anchors[:, 2])
                th = torch.log((pos_preds[:, 3] + epsilon) / pos_anchors[:, 3])
                t = torch.stack((tx, ty, tw, th))

                matched_bboxes = gt_bboxes[gt_bboxes_indices].double().to(device)
                gtx = matched_bboxes[:, 0] - pos_anchors[:, 0]
                gty = matched_bboxes[:, 1] - pos_anchors[:, 1]
                gtw = torch.log(matched_bboxes[:, 2] / pos_anchors[:, 2])
                gth = torch.log(matched_bboxes[:, 3] / pos_anchors[:, 3])
                gt = torch.stack((gtx, gty, gtw, gth))

                pos_targets = torch.zeros_like(pos_preds)
                pos_targets[:, 1] = 1.0
                neg_targets = torch.zeros_like(neg_preds)
                neg_targets[:, 0] = 1.0

                effective_preds = torch.cat((pos_preds, neg_preds))
                targets = torch.cat((pos_targets, neg_targets))

                shuffle_indexes = torch.randperm(effective_preds.size()[0])
                effective_preds = effective_preds[shuffle_indexes]
                targets = targets[shuffle_indexes]

                loss_class = F.binary_cross_entropy(F.sigmoid(effective_preds), targets)
                loss_reg = F.smooth_l1_loss(t, gt)

                loss = loss_class + loss_reg

                print("loss_class {}, loss_reg {}".format(
                    loss_class, loss_reg
                ))

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def persist(self, is_best=False):
        model_dir = os.path.join(self.log_dir, 'models')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        file_name = (
            "epoch_{}_best.pth.tar" if is_best else "epoch_{}.pth.tar") \
            .format(self.current_epoch)

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        state_path = os.path.join(model_dir, file_name)
        torch.save(state, state_path)
