import logging
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn

from anchor import generate_anchors, mark_anchors
from config import Config
from utils import change_coordinate, seek_model, change_coordinate_inv
from logger import Logger
from evaluate import evaluate

device = torch.device(Config.DEVICE)


class Trainer(object):

    def __init__(self, optimizer, model, training_dataloader,
                 validation_dataloader, log_dir=False, max_epoch=100,
                 resume=False, persist_stride=1, verbose=False):

        self.start_epoch = 1
        self.current_epoch = 1

        self.verbose = verbose
        self.max_epoch = max_epoch
        self.persist_stride = persist_stride

        # initialize log
        self.log_dir = log_dir
        log_file = os.path.join(self.log_dir, 'log.txt')
        logging.basicConfig(filename=log_file, level=logging.DEBUG)
        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # initialize tensorboard
        if Config.TENSOR_BOARD_ENABLED:
            tensor_board_dir = os.path.join(self.log_dir, 'tensorboard')
            if not os.path.isdir(tensor_board_dir):
                os.mkdir(tensor_board_dir)
            self.logger = Logger(tensor_board_dir)

        # initialize model
        self.optimizer = optimizer
        self.model = model.float()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

        self.model.load_state_dict(model_zoo.load_url(Config.VGG16_PRETRAINED_WEIGHTS), strict=False)
        self.resume = str(resume) if resume else False

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        # initialize anchors
        self.anchors = np.vstack(
            list(map(lambda x: np.array(x), generate_anchors(
                Config.ANCHOR_STRIDE,
                Config.ANCHOR_SIZE,
                (Config.IMAGE_SIZE,) * 2
            )))
        )
        self.anchor_coord_changed = change_coordinate_inv(self.anchors)
        self.len_anchors = len(self.anchors)

        # resume from some model
        if self.resume:
            state_file = seek_model(self.resume)

            print("loading checkpoint {}".format(state_file))
            checkpoint = torch.load(state_file)
            self.start_epoch = self.current_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint {} (epoch {})".format(
                state_file, self.current_epoch))

    def train(self):
        for self.current_epoch in range(self.start_epoch, self.max_epoch + 1):
            self.run_epoch(mode='train')
            if not (self.current_epoch % self.persist_stride):
                self.persist()
            if self.validation_dataloader:
                self.run_epoch(mode='validate')

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

            for index, (images, all_gt_bboxes, path) in enumerate(dataloader):
                if mode == 'validate':
                    break
                # gt_bboxes: 2-d list of (batch_size, ndarray(bbox_size, 4) )
                image = images.float().permute(0, 3, 1, 2).to(device)
                res = self.model(image)

                predictions = list(zip(*list(self.model(image))))
                # get and flatten reg_preds and cls_preds from predictions
                reg_preds_list = []
                cls_preds_list = []
                for i, prediction in enumerate(predictions):
                    prediction = list(prediction)
                    for k, feature_map_prediction in enumerate(prediction):
                        prediction[k] = feature_map_prediction \
                            .view(feature_map_prediction.size()[0], -1) \
                            .permute(1, 0).contiguous()
                    reg_preds_list.append(torch.cat(prediction[::2]))
                    cls_preds_list.append(torch.cat(prediction[1::2]))

                total_t = []
                total_gt = []
                total_effective_pred = []
                total_target = []

                for i, reg_preds in enumerate(reg_preds_list):
                    cls_preds = cls_preds_list[i]
                    gt_bboxes = all_gt_bboxes[i]

                    if len(gt_bboxes) == 0:
                        # no ground truth bounding boxes, ignored
                        continue

                    pos_indices, gt_bboxes_indices, neg_indices = \
                        mark_anchors(self.anchor_coord_changed, gt_bboxes,
                                     positive_threshold=Config.POSITIVE_ANCHOR_THRESHOLD,
                                     negative_threshold=Config.NEGATIVE_ANCHOR_THRESHOLD,
                                     least_pos_num=Config.LEAST_POSITIVE_ANCHOR_NUM)

                    # in very rare case of no positive anchors
                    if len(pos_indices) == 0:
                        continue

                    # make samples of negative to positive 3:1
                    n_neg_indices = len(pos_indices) * Config.NEG_POS_ANCHOR_NUM_RATIO

                    # hard neg example mining
                    neg_cls_preds = cls_preds[neg_indices]
                    neg_indices = torch.sort(neg_cls_preds[:, 0])[1][:n_neg_indices]

                    pos_anchors = torch.tensor(
                        self.anchors[pos_indices]
                    ).float().to(device)

                    # preds bbox is tx ty tw th
                    total_t.append(reg_preds[pos_indices])

                    gt_bboxes = change_coordinate(gt_bboxes)
                    gt_bboxes = torch.tensor(gt_bboxes).float().to(device)
                    matched_bboxes = gt_bboxes[gt_bboxes_indices]

                    gtx = (matched_bboxes[:, 0] - pos_anchors[:, 0]) / pos_anchors[:, 2]
                    gty = (matched_bboxes[:, 1] - pos_anchors[:, 1]) / pos_anchors[:, 3]
                    gtw = torch.log(matched_bboxes[:, 2] / pos_anchors[:, 2])
                    gth = torch.log(matched_bboxes[:, 3] / pos_anchors[:, 3])
                    gt = torch.stack((gtx, gty, gtw, gth), dim=1)
                    total_gt.append(gt)

                    pos_targets = torch.ones(len(pos_indices)).long().to(device)
                    neg_targets = torch.zeros(len(neg_indices)).long().to(device)

                    effective_preds = torch.cat((cls_preds[pos_indices], neg_cls_preds[neg_indices]))
                    targets = torch.cat((pos_targets, neg_targets))

                    shuffle_indexes = torch.randperm(effective_preds.size()[0])
                    effective_preds = effective_preds[shuffle_indexes]
                    targets = targets[shuffle_indexes]
                    total_effective_pred.append(effective_preds)
                    total_target.append(targets)

                # in very rare case of no positive anchors
                if len(total_t) == 0:
                    continue

                total_t = torch.cat(total_t)
                total_gt = torch.cat(total_gt)
                total_targets = torch.cat(total_target)
                total_effective_pred = torch.cat(total_effective_pred)

                loss_class = F.cross_entropy(
                    total_effective_pred, total_targets,
                )
                loss_reg = F.smooth_l1_loss(total_t, total_gt)
                loss = loss_class + loss_reg

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_class_loss += loss_class.data
                total_reg_loss += loss_reg.data
                total_loss += loss.data

                if not index % Config.LOSS_LOG_STRIDE:
                    logging.info(
                        "[{}][epoch:{}][iter:{}][total:{}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                            mode, self.current_epoch, index, total_iter, loss_class.data, loss_reg.data, loss.data
                        )
                    )

                    if Config.TENSOR_BOARD_ENABLED and mode == 'train':
                        info = {
                            'train_loss_classification': loss_class.data,
                            'train_loss_regression': loss_reg.data,
                            'train_total_loss': loss.data,
                        }

                        for tag, value in info.items():
                            step = (self.current_epoch-1) * total_iter + index
                            self.logger.scalar_summary(tag, value, step)

            if Config.TENSOR_BOARD_ENABLED and mode == 'train':
                # Log the scalar values

                logging.info('[{}][epoch:{}] total_class_loss - {} total_reg_loss {} - total_loss {}'.format(
                    mode, self.current_epoch, total_class_loss / total_iter, total_reg_loss / total_iter, total_loss / total_iter
                ))

                info = {
                    'average_train_loss_classification': total_class_loss / total_iter,
                    'average_train_loss_regression': total_reg_loss / total_iter,
                    'average_train_total_loss': total_loss / total_iter,
                }

                for tag, value in info.items():
                    step = self.current_epoch
                    self.logger.scalar_summary(tag, value, step)

            elif Config.TENSOR_BOARD_ENABLED and mode == 'validate':
                # compute mAP
                logging.info('[epoch:{}] computing mAP...'.format(self.current_epoch))
                mAP = evaluate(self.model)
                logging.info('[epoch:{}] mAP is {}'.format(self.current_epoch, mAP))
                self.logger.scalar_summary('mean_average_precision', mAP, self.current_epoch)

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
