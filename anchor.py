import numpy as np

def generate_anchors(anchor_stride=[4, 8, 16, 32, 64, 128],
                     anchor_size=[16, 32, 64, 128, 256, 512],
                     image_size=640):
    all_anchors = []

    for i in range(len(anchor_stride)):
        anchors = []

        stride = anchor_stride[i]
        size = anchor_size[i]

        for row in range(image_size // stride):
            top = row * stride
            for col in range(image_size // stride):
                left = col * stride
                bottom = top + size
                right = left + size
                anchors.append((top, left, bottom, right))

        all_anchors.append(anchors)

    return all_anchors


def mark_anchors(anchors, gt_boxes, positive_threshold=0.5,
                 negative_threshold=0.3):
    """IoU larger than positive_threshold is positive anchors,
    less than negative_threshold is negative anchors
    """
    iou = compute_iou(anchors, gt_boxes)
    max_iou = iou.max(axis=1)

    positive_anchor_indices = np.where(max_iou > positive_threshold)
    negative_anchor_indices = np.where(max_iou < negative_threshold)

    # positive anchors should get coorsponding gt_boxes for computing deltas
    positive_iou = iou[max_iou > positive_threshold]
    matched_gt_box_indices = positive_iou.argmax(axis=1)

    return positive_anchor_indices, matched_gt_box_indices, negative_anchor_indices


def compute_iou(anchors, gt_boxes):
    """compute IoU for 2 bounding boxes arrays, return size is:
    [anchors.shape[0], gt_boxes[0]]
    """
    len_anchors = anchors.shape[0]
    len_gt_boxes = gt_boxes.shape[0]
    anchors = np.repeat(anchors, len_gt_boxes, axis=0)
    gt_boxes = np.vstack([gt_boxes] * len_anchors)

    y1 = np.maximum(anchors[:, 0], gt_boxes[:, 0])
    x1 = np.maximum(anchors[:, 1], gt_boxes[:, 1])
    y2 = np.minimum(anchors[:, 2], gt_boxes[:, 2])
    x2 = np.minimum(anchors[:, 3], gt_boxes[:, 3])

    y_zeros = np.zeros_like(y2.shape)
    x_zeros = np.zeros_like(x2.shape)

    intersect = np.maximum((y2 - y1), y_zeros) * np.maximum((x2 - x1), x_zeros)

    unit = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]) + \
           (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) - \
           intersect

    return (intersect / unit).reshape(len_anchors, len_gt_boxes)

if __name__ == "__main__":
    anchors = np.array((
        (1, 1, 3, 3),
        (3, 4, 6, 7),
        (4, 3, 6, 6)
    ))

    gt_boxes = np.array((
        (2, 2, 5, 5),
        (4, 3, 6, 5),
        (4, 3, 6, 5)
    ))

    print(mark_anchors(anchors, gt_boxes))

