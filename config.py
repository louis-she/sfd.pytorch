class Config(object):

    DEVICE = 'cuda'

    # pathes
    DATASET_DIR = '/home/louis/datasets/wider_face'
    LOG_DIR = '/media/louis/ext4/models/sfd'

    # training && log controls
    MODEL_SAVE_STRIDE = 1
    BATCH_SIZE = 8
    RESUME_FROM = False  # epoch number, model file name or path are all OK
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    POSITIVE_ANCHOR_THRESHOLD = 0.5
    NEGATIVE_ANCHOR_THRESHOLD = 0.1
    LEAST_POSITIVE_ANCHOR_NUM = 50
    LOSS_LOG_STRIDE = 1  # log loss every N iter
    DATALOADER_WORKER_NUM = 4
    VGG16_PRETRAINED_WEIGHTS = "https://download.pytorch.org/models/vgg16-397923af.pth"

    # anchors, have skipped the first feature map cause I'm
    # not very interested at very tiny faces
    IMAGE_SIZE = 640
    ANCHOR_STRIDE = [8, 16, 32, 64, 128]
    ANCHOR_SIZE = [32, 64, 128, 256, 512]
    NEG_POS_ANCHOR_NUM_RATIO = 3
    
    # nms threshold
    NMS_THRESHOLD = 0.3