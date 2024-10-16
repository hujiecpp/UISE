from detectron2.config import CfgNode as CN

def add_uise_config(cfg):
    cfg.MODEL.UISE = CN()
    cfg.MODEL.UISE.SIZE_DIVISIBILITY = 32
    cfg.MODEL.UISE.NUM_CLASSES = 133
    cfg.MODEL.UISE.NUM_STAGES = 2
    
    cfg.MODEL.UISE.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.UISE.HIDDEN_DIM = 256
    cfg.MODEL.UISE.AGG_DIM = 128
    cfg.MODEL.UISE.NUM_PROPOSALS = 100
    cfg.MODEL.UISE.CONV_KERNEL_SIZE_2D = 1
    cfg.MODEL.UISE.CONV_KERNEL_SIZE_1D = 3
    cfg.MODEL.UISE.NUM_CLS_FCS = 1
    cfg.MODEL.UISE.NUM_MASK_FCS = 1
    cfg.MODEL.UISE.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.UISE.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.UISE.CLASS_WEIGHT = 2.0
    cfg.MODEL.UISE.MASK_WEIGHT = 5.0
    cfg.MODEL.UISE.DICE_WEIGHT = 5.0
    cfg.MODEL.UISE.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.UISE.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.UISE.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.UISE.TEMPERATIRE = 0.1

    cfg.MODEL.UISE.TEST = CN()
    cfg.MODEL.UISE.TEST.SEMANTIC_ON = False
    cfg.MODEL.UISE.TEST.INSTANCE_ON = False
    cfg.MODEL.UISE.TEST.PANOPTIC_ON = False
    cfg.MODEL.UISE.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.UISE.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.UISE.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = None
    
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0

    # cfg.INPUT.DATASET_MAPPER_NAME = "yoso_panoptic_lsj"
    cfg.INPUT.DATASET_MAPPER_NAME = "yoso_instance_lsj"
    # cfg.INPUT.DATASET_MAPPER_NAME = "yoso_semantic_lsj"
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0

    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

def add_uise_video_config(cfg):
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = []
