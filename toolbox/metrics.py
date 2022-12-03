# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import torch
from torchvision import transforms

class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        # self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.iou = 0.0
        self.ber = 0.0
        self.mae = 0.0
        self.img_num = 0

        # self.iou1 = 0.0
        # self.ber1 = 0.0
        # self.mae1 = 0.0

        # if ignore_index is None or ignore_index < 0 or ignore_index > n_classes:
        #     self.ignore_index = None
        # elif isinstance(ignore_index, int):
        #     self.ignore_index = (ignore_index,)
        # else:
        #     try:
        #         self.ignore_index = tuple(ignore_index)
        #     except TypeError:
        #         raise ValueError("'ignore_index' must be an int or iterable")

    # def _fast_hist(self, label_true, label_pred, n_class):
    #     mask = (label_true >= 0) & (label_true < n_class)
    #     hist = np.bincount(
    #         n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
    #     ).reshape(n_class, n_class)
    #     return hist

    def compute_iou(self, label_trues, label_preds):
        """
        (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
        Here, n_cl = 1 as we have only one class (mirror).
        """

        if np.sum(label_preds) == 0 or np.sum(label_trues) == 0:
            iou_ = 0
            return iou_

        n_ii = np.sum(np.logical_and(label_preds, label_trues))
        t_i = np.sum(label_trues)
        n_ij = np.sum(label_preds)

        iou_ = n_ii / (t_i + n_ij - n_ii)

        return iou_

    def compute_mae(self, label_trues, label_preds):

        N_p = np.sum(label_trues)
        N_n = np.sum(np.logical_not(label_trues))

        mae_ = np.mean(abs(label_preds - label_trues)).item()

        return mae_

    def compute_ber(self, label_trues, label_preds):


        N_p = np.sum(label_trues)
        N_n = np.sum(np.logical_not(label_trues))

        TP = np.sum(np.logical_and(label_preds, label_trues))
        TN = np.sum(np.logical_and(np.logical_not(label_preds), np.logical_not(label_trues)))

        ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

        return ber_

    # def compute_iou1(self, label_trues, label_preds):
    #     """
    #     (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    #     Here, n_cl = 1 as we have only one class (mirror).
    #     """
    #     # label_preds = torch.from_numpy(label_preds).type(torch.FloatTensor)
    #     # label_trues = torch.from_numpy(label_trues).type(torch.FloatTensor)
    #
    #     trans = transforms.Compose([transforms.ToTensor()])
    #     label_preds = trans(label_preds).cuda()
    #     label_trues = trans(label_trues).cuda()
    #     label_preds = (label_preds >= 0.5)
    #     label_trues = (label_trues >= 0.5)
    #
    #     iou = torch.sum((label_preds & label_trues)) / torch.sum((label_preds | label_trues))
    #
    #     return iou.item()
    #
    # def compute_mae1(self, label_trues, label_preds):
    #
    #     # label_preds = torch.from_numpy(label_preds).type(torch.FloatTensor)
    #     # label_trues = torch.from_numpy(label_trues).type(torch.FloatTensor)
    #
    #     trans = transforms.Compose([transforms.ToTensor()])
    #     label_preds = trans(label_preds).cuda()
    #     # print("333", label_trues, label_trues.shape)
    #     label_trues = trans(label_trues).cuda()
    #
    #     label_preds = torch.where(label_preds >= 0.5, torch.ones_like(label_preds), torch.zeros_like(label_preds))
    #     label_trues = torch.where(label_trues >= 0.5, torch.ones_like(label_trues), torch.zeros_like(label_trues))
    #
    #     # pred = 1 - pred
    #     # gt = 1 - gt
    #
    #     mae = torch.abs(label_preds - label_trues).float().mean()
    #
    #     return mae.item()
    #
    # def compute_ber1(self, label_trues, label_preds):
    #
    #     # label_preds = torch.from_numpy(label_preds).type(torch.FloatTensor)
    #     # label_trues = torch.from_numpy(label_trues).type(torch.FloatTensor)
    #
    #     trans = transforms.Compose([transforms.ToTensor()])
    #     label_preds = trans(label_preds).cuda()
    #     # print("444", label_trues, label_trues.shape)
    #     label_trues = trans(label_trues).cuda()
    #
    #     label_preds = (label_preds >= 0.5)
    #     label_trues = (label_trues >= 0.5)
    #
    #     N_p = torch.sum(label_trues) + 1e-20
    #     N_n = torch.sum(torch.logical_not(label_trues)) + 1e-20  # should we add this？
    #
    #     TP = torch.sum(label_preds & label_trues)
    #     TN = torch.sum(torch.logical_not(label_preds) & torch.logical_not(label_trues))
    #
    #     ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))
    #
    #     return ber

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            # print(lt.shape, lp.shape)
            # self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            self.iou += self.compute_iou(lt, lp)
            self.ber += self.compute_ber(lt, lp)
            self.mae += self.compute_mae(lt, lp)
            self.img_num += 1
            #
            # self.iou1 += self.compute_iou1(lt, lp)
            # self.ber1 += self.compute_ber1(lt, lp)
            # self.mae1 += self.compute_mae1(lt, lp)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        # hist = self.confusion_matrix
        iou = self.iou
        ber = self.ber * 100
        mae = self.mae
        img_num =self.img_num

        # iou1 = self.iou1
        # ber1 = self.ber1 * 100
        # mae1 = self.mae1

        ##ignore unlabel
        # if self.ignore_index is not None:
        #     for index in self.ignore_index:
        #         hist = np.delete(hist, index, axis=0)
        #         hist = np.delete(hist, index, axis=1)
        #
        # acc = np.diag(hist).sum() / hist.sum()
        # cls_acc = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(cls_acc)
        # iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # mean_iou = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # # set unlabel as nan
        # if self.ignore_index is not None:
        #     for index in self.ignore_index:
        #         iu = np.insert(iu, index, np.nan)
        #
        # cls_iu = dict(zip(range(self.n_classes), iu))
        # cls_acc = dict(zip(range(self.n_classes), cls_acc))

        iou /= img_num
        ber /= img_num
        mae /= img_num

        # iou1 /= img_num
        # ber1 /= img_num
        # mae1 /= img_num

        return (
            {
                # "pixel_acc: ": acc,
                # "class_acc: ": acc_cls,
                # "mIou: ": mean_iou,
                # "fwIou: ": fw_iou,
                "iou1: ": iou,
                "ber1: ": ber,
                "mae1: ": mae,
                # "iou4: ": iou1,
                # "ber4: ": ber1,
                # "mae4: ": mae1,
            },
            # cls_iu,
            # cls_acc,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
