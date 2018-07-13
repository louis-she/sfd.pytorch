import numpy as np
from anchor import compute_iou


def AP(prediction, gt, iou_threshold):
    """compute average precision of detection, all the coordinate should be
    (top left bottom right)

    Args:
        predict_bboxes (ndarray): should be a N * (4 + K) ndarray
            N is number of boxes been predicted(batch_size),
            4 represents [top, left, bottom, right],
            K is the score of every class, len(K) equals to class number
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
    scores = softmax(prediction[:, 4:])
    klasses = np.argmax(scores, axis=1)
    scores = scores[np.arange(len(scores)), klasses]

    # sort klass, scores, bboxes by value of scores
    sorted_indices = np.argsort(scores)[::-1]
    scores = scores[sorted_indices]
    klasses = klasses[sorted_indices]
    bboxes = bboxes[sorted_indices]

    # get a list `result` of tp and fp, length should be the same as bboxes
    result = np.zeros(len(bboxes))
    matched_index = []
    ious = compute_iou(bboxes, gt[:, :4])
    for index, iou in enumerate(ious):
        gt_index = np.argmax(iou)
        if iou[gt_index] > iou_threshold and gt_index not in matched_index \
                and gt_index == gt[gt_index, 5]:
            result[index] = 1
            matched_index.append(gt_index)

    # get tp and fp result of every class
    ap_of_klass = {}
    for klass in np.unique(klasses):
        klass_indices = klasses == klass
        klass_result = result[klass_indices]

        # the following block can be replaced by
        # `sklearn.metrics.average_precision_score`
        cumsum = np.cumsum(klass_result)
        recall_point = np.unique(cumsum)
        precisions = np.zeros_like(recall_point, dtype=np.float)
        for tp_num in recall_point:
            predictions_num = np.searchsorted(cumsum, tp_num) + 1.0
            precisions[tp_num - 1] = float(tp_num) / predictions_num
        ap = np.sum(precisions) / len(precisions)
        ap_of_klass[klass] = ap

    return ap_of_klass


def softmax(mat):
    """change a vector to softmax score in batch

    Args:
        mat (ndarray): 2 dimensional matrix, shape is [batch_size, array_size]

    Returns:
        ndarray: a tensor which is has the same shape as the input
    """
    mat_exp = np.exp(mat)
    mat_sum = np.sum(mat_exp, axis=1, keepdims=True)
    return mat_exp / mat_sum
