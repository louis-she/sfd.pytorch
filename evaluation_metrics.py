import numpy as np
from anchor import compute_iou
import torch


def AP(prediction, gt, iou_threshold):
    """compute average precision of detection, all the coordinate should be
    (top left bottom right)

    Args:
        predict_bboxes (ndarray): should be a N * (4 + 1 + 1) ndarray
            N is number of boxes been predicted(batch_size),
            4 represents [top, left, bottom, right],
            1 is the confidence of the class
            1 is the number represents the class
        gt_bboxes (ndarray): should be a M * (4 + 1) ndarray
            M is the number of ground truth bboxes of that image
            4 represents [top, left, bottom, right],
            1 represents the class number of the bbox. Since we use 0 to be the
                background, so class number of object should be started from 1
        iou_threshold (float): threshold of iou for seperate the true positive
            or false positive
        num_classes (int): how many classes of the target
    Returns: vector of class_number size, each element is AP
        value of every class
    """
    # apply softmax for prediction[:, 4:], get the highest index and klass
    bboxes = prediction[:, :4]
    scores = prediction[:, 4]
    klasses = prediction[:, 5]

    # sort klass, scores, bboxes by value of scores
    inds = np.argsort(scores)[::-1]
    scores, klasses, bboxes = scores[inds], klasses[inds], bboxes[inds]

    # get a list result of tp and fp, length should be the same as bboxes
    result = np.zeros(len(bboxes))
    matched_index = []

    ious = compute_iou(bboxes, gt[:, :4])
    for index, iou in enumerate(ious):
        gt_index = np.argmax(iou)
        if iou[gt_index] > iou_threshold \
                and gt_index not in matched_index \
                and klasses[index] == gt[gt_index, 4]:
            result[index] = 1
            matched_index.append(gt_index)

    # get tp and fp result of every class
    ap_of_klass = {}
    for klass in np.unique(klasses):
        klass_indices = klasses == klass
        klass_result = result[klass_indices]

        object_num = np.sum(gt[:, 4] == klass)

        cumsum = np.cumsum(klass_result)
        recall_point_num = np.unique(cumsum)
        precisions = np.zeros_like(recall_point_num, dtype=np.float)
        recalls = np.zeros_like(recall_point_num, dtype=np.float)

        for recall_point in recall_point_num:
            recall_point = int(recall_point)
            if recall_point == 0:
                continue
            predictions_num = np.searchsorted(cumsum, recall_point) + 1.0
            precisions[recall_point - 1] = float(recall_point) / predictions_num
            recalls[recall_point - 1] = recall_point / object_num

        recalls = np.insert(recalls, 0, 0.0)
        precisions = np.insert(precisions, 0, 0.0)
        recalls = np.append(recalls, 1.0)
        precisions = np.append(precisions, 0.0)

        # make precision monotone decreased
        current_precision = 0
        for i in range(len(precisions) - 1, -1, -1):
            precisions[i] = max(current_precision, precisions[i])
            current_precision = precisions[i]

        ap = 0
        for i in range(1, len(precisions)):
            precision = precisions[i]
            recall_span = recalls[i] - recalls[i - 1]
            ap += precision * recall_span

        ap_of_klass[klass] = ap

    return ap_of_klass


def softmax(mat):
    """change a vector to softmax score in batch

    Args:
        mat (ndarray): 2 dimensional matrix, shape is [batch_size, array_size]

    Returns:
        ndarray: a tensor which is has the same shape as the input
    """
    mat_exp = torch.exp(mat)
    mat_sum = torch.sum(mat_exp, dim=1, keepdim=True)
    return mat_exp / mat_sum
