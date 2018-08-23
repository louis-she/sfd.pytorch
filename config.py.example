class Config(object):

    DEVICE = 'cuda'

    # pathes
    VOC_DATASET_DIR = '/home/xxx/datasets/VOC2012'
    WF_DATASET_DIR = '/home/xxx/datasets/wider_face'
    LOG_DIR = '/home/xxx/datasets/logs/'

    # datasets
    DATASETS = 'WF' # currently support 'WF' and 'VOC'

    # VOC datasets utilities
    VOC_CLASS = 'person'

    # training && log controls
    MODEL_SAVE_STRIDE = 1
    BATCH_SIZE = 4
    RESUME_FROM = False  # epoch number, model file name or path are all OK
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.9
    EPOCHS = 500

    POSITIVE_ANCHOR_THRESHOLD = 0.3
    NEGATIVE_ANCHOR_THRESHOLD = 0.1
    LEAST_POSITIVE_ANCHOR_NUM = 100
    LOSS_LOG_STRIDE = 1  # log loss every N iter
    DATALOADER_WORKER_NUM = 1
    VGG16_PRETRAINED_WEIGHTS = "https://download.pytorch.org/models/vgg16-397923af.pth"

    """
    image augmentation
    """
    # relative to the shorter edge of the image
    MIN_CROPPED_RATIO = 0.3
    MAX_CROPPED_RATIO = 1
    # if KEEP_THRESHOLD area or larger of ground truth bounding is in the
    # cropped image, then keep keep and crop the ground truth bounding box.
    KEEP_AREA_THRESHOLD = 0.5

    RANDOM_FLIP = True
    RANDOM_COLOR_JITTER = False

    # anchors, have skipped the first feature map cause I'm
    # not very interested at very tiny faces
    IMAGE_SIZE = 640
    ANCHOR_STRIDE = [4, 8, 16, 32, 64, 128]
    ANCHOR_SIZE = [16, 32, 64, 128, 256, 512]
    NEG_POS_ANCHOR_NUM_RATIO = 3

    # nms threshold
    NMS_THRESHOLD = 0.3
    PREDICTION_THRESHOLD = 0.8

    # tensorboard
    TENSOR_BOARD_ENABLED = True  # if enabled, the tensorflow log dir is in
                                 # $LOG_DIR/tensorboad
