def loss_function(anchors, predictions, ground_truth):
    """compute loss and return
    Args:
        predictions: the return value from model, should be
            (batch_size, 4+2)
        ground_truth: (batch_size, face_bounding_box_size, 4)
    """

