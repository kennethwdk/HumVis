import os

from yacs.config import CfgNode as CN

_C = CN()

_C.CFG_NAME = ''
_C.OUTPUT_DIR = ''
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.DIST_BACKEND = 'nccl'
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.VERBOSE = True
_C.DDP = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'CID'
_C.MODEL.DEVICE = 'cpu'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.SYNC_BN = False
_C.MODEL.BACKBONE = 'HRNet-W32'
_C.MODEL.BIAS_PROB = 0.01
_C.MODEL.IIA = CN()
_C.MODEL.IIA.IN_CHANNELS = 480
_C.MODEL.IIA.OUT_CHANNELS = 18
_C.MODEL.GFD = CN()
_C.MODEL.GFD.IN_CHANNELS = 480
_C.MODEL.GFD.CHANNELS = 32
_C.MODEL.GFD.OUT_CHANNELS = 17

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'data'
_C.DATASET.DATASET = 'ochuman'
_C.DATASET.NUM_KEYPOINTS = 17
_C.DATASET.MAX_INSTANCES = 100
_C.DATASET.TRAIN = 'val'
_C.DATASET.TEST = 'test'
_C.DATASET.FILTER_IMAGE = False
_C.DATASET.SIGMA = 2.0
_C.DATASET.FLIP = 0.5
_C.DATASET.FLIP_INDEX = []

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = 128

# testing
_C.TEST = CN()
_C.TEST.FLIP_TEST = False
_C.TEST.IMAGES_PER_GPU = 1
_C.TEST.MODEL_FILE = 'model_files/cid32.tar'
_C.TEST.OKS_SCORE = 0.7
_C.TEST.OKS_SIGMAS = [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]

_C.TEST.MAX_PROPOSALS = 30
_C.TEST.KEYPOINT_THRESHOLD = 0.2
_C.TEST.CENTER_POOL_KERNEL = 3

_C.TEST.POOL_THRESHOLD1 = 300
_C.TEST.POOL_THRESHOLD2 = 200