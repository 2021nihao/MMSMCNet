from .metrics import averageMeter, runningScore
from .log import get_logger
import torch.nn as nn
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .losses.loss import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LovaszSoftmax, LDAMLoss, MscCrossEntropyLoss

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'pst900':
        from .datasets.mirrorrgbd import MirrorRGBD
        return pst900(cfg, mode='train'), MirrorRGBD(cfg, mode='test')





def get_model(cfg):

    if cfg['model_name'] == "MMSMCNet":
        from toolbox.models.MMSMCNet import nation
        return nation()

   

