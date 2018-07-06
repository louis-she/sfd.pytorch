import os
import sys
import logging
from time import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import change_coordinate
from anchor import generate_anchors, mark_anchors

device = torch.device("cuda")

class Trainer(object):

    def __init__(self, optimizer, model, training_dataloader,
                 validation_dataloader, log_dir=False, max_epoch=100,
                 resume=False, persist_stride=1, verbose=False):

        # debug
        self.verbose = verbose
        self.log_dir = log_dir
        log_file = os.path.join(self.log_dir, 'log.txt')
        logging.basicConfig(filename=log_file, level=logging.DEBUG)

        self.optimizer = optimizer
        self.model = model.float().to(device)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.max_epoch = max_epoch
        if resume:
            self.resume = str(resume)
        else:
            self.resume = False
        self.persist_stride = persist_stride
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.start_epoch = 1
        self.current_epoch = 1
        self.anchors = np.vstack(
            list(map(lambda x: np.array(x), generate_anchors()))
        )
        self.len_anchors = len(self.anchors)

        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if self.resume:
            candidate_a = os.path.join(self.log_dir, 'models', self.resume)
            candidate_b = os.path.join(self.log_dir, 'models', 'epoch_{}.pth.tar'.format(self.resume))
            candidate_c = self.resume

            if os.path.isfile(candidate_a):
                state_file = candidate_a
            elif os.path.isfile(candidate_b):
                state_file = candidate_b
            elif os.path.isfile(candidate_c):
                state_file = candidate_c
            else:
                raise RuntimeError(
                    "resume file {} is not found".format(resume)
                )

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

    def mark(self, checkpoint, reset=False):
        if self.verbose:
            torch.cuda.synchronize()
            if reset:
                self.execute_time = time()
                self.time_log = []
            self.time_log.append((checkpoint, time() - self.execute_time))

    def execute_info(self):
        if self.verbose:
            last_time = 0
            for log in self.time_log:
                current_time = log[1]
                checkpoint = log[0]
                time_span = current_time - last_time
                print("{} ==> {}".format(time_span, checkpoint))
                last_time = current_time

    def run_epoch(self, mode):
        if mode == 'train':
            dataloader = self.training_dataloader
            self.model.train()
        else:
            dataloader = self.validation_dataloader
            self.model.eval()

        with torch.set_grad_enabled(mode == 'train'):
            total_class_loss = 0
            total_reg_loss = 0
            total_loss = 0
            total_iter = len(dataloader)

            for index, (images, all_gt_bboxes, pathes) in enumerate(dataloader):
                # gt_bboxes: 2-d list of (batch_size, ndarray(bbox_size, 4) )
                image = images.permute(0, 3, 1, 2).float().to(device)
                predictions = list(self.model(image))
                predictions = list(zip(*predictions))
                for i, prediction in enumerate(predictions):
                    prediction = list(prediction)
                    for k, feature_map_prediction in enumerate(prediction):
                        prediction[k] = feature_map_prediction.view(6, -1).permute(1, 0)
                    # prediction is 6 x N x 6
                    predictions[i] = torch.cat(prediction)

                # predictions elements is 6 x anchor_size(34125 this case)

                total_t = []
                total_gt = []
                total_effective_pred = []
                total_target = []
                for i, prediction in enumerate(predictions):
                    gt_bboxes = all_gt_bboxes[i]
                    # pick up positive and negative anchors
                    pos_indices, gt_bboxes_indices, neg_indices = \
                        mark_anchors(self.anchors, gt_bboxes)

                    # in case of no positive anchors
                    if len(pos_indices) == 0:
                        continue

                    # make samples of negative to positive 3:1
                    n_neg_indices = len(pos_indices) * 3
                    reg_random_indices = torch.randperm(len(neg_indices))
                    neg_indices = neg_indices[reg_random_indices][:n_neg_indices]

                    pos_anchors = torch.tensor(
                        change_coordinate(self.anchors[pos_indices])
                    ).float().to(device)

                    neg_anchors = torch.tensor(
                        change_coordinate(self.anchors[neg_indices])
                    ).float().to(device)

                    pos_preds = prediction[pos_indices]
                    neg_preds = prediction[neg_indices]

                    # preds bbox is tx ty tw th
                    total_t.append(pos_preds[:, :4])

                    gt_bboxes = change_coordinate(gt_bboxes)
                    gt_bboxes = torch.tensor(gt_bboxes).float().to(device)
                    matched_bboxes = gt_bboxes[gt_bboxes_indices]
                    gtx = matched_bboxes[:, 0] - pos_anchors[:, 0]
                    gty = matched_bboxes[:, 1] - pos_anchors[:, 1]
                    gtw = torch.log(matched_bboxes[:, 2] / pos_anchors[:, 2])
                    gth = torch.log(matched_bboxes[:, 3] / pos_anchors[:, 3])
                    gt = torch.stack((gtx, gty, gtw, gth), dim=1)
                    total_gt.append(gt)

                    pos_targets = torch.ones(pos_preds.size()[0]).long().to(device)
                    neg_targets = torch.zeros(neg_preds.size()[0]).long().to(device)

                    effective_preds = torch.cat((pos_preds[:, 4:], neg_preds[:, 4:]))
                    targets = torch.cat((pos_targets, neg_targets))

                    shuffle_indexes = torch.randperm(effective_preds.size()[0])
                    effective_preds = effective_preds[shuffle_indexes]
                    targets = targets[shuffle_indexes]
                    total_effective_pred.append(effective_preds)
                    total_target.append(targets)

                total_t = torch.cat(total_t)
                total_gt = torch.cat(total_gt)
                total_targets = torch.cat(total_target)
                total_effective_pred = torch.cat(total_effective_pred)

                loss_class = 10 * F.cross_entropy(
                    total_effective_pred, total_targets
                )
                loss_reg = F.smooth_l1_loss(total_t, total_gt)
                loss = loss_class + loss_reg

                total_class_loss += loss_class.data
                total_reg_loss += loss_reg.data
                total_loss += loss.data

                if not index % 1:
                    logging.info(
                        "[{}][epoch:{}][iter:{}][total:{}] loss_class {:.8f} - loss_reg {:.8f}".format(
                            mode, self.current_epoch, index, total_iter, loss_class, loss_reg
                        )
                    )

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            logging.info('[{}][epoch:{}] total_class_loss - {} total_reg_loss {} - total_loss {}'.format(
                mode, self.current_epoch, total_class_loss / total_iter, total_reg_loss / total_iter, total_loss / total_iter
            ))

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
