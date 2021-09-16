import torch
import torch.nn as nn
from .config import DefaultConfig


def fmap_to_og_coords(feature, stride):
    """ transform a feature map's coordinates into original coordinates """
    h, w = feature.shape[1: 3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])

    coords = torch.stack([shift_x, shift_y], -1) + stride // 2

    return coords


class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        """
        inputs: a list

        index 0: list [cls_logits, ctr_logits, reg_preds]
        cls_logits: [batch_size, class_num, h, w]
        ctr_logits: [batch_size, 1, h, w]
        reg_preds: [batch_size, 4, h, w]

        index 1: gt_boxes [batch_size, m, 4] tensor of floats

        index 2: classes [batch_size, m] tensor of longs

        returns:

        cls_targets: [batch_size, sum(_h * _w), 1]
        ctr_targets: [batch_size, sum(_h * _w), 1]
        reg_targets: [batch_size, sum(_h * _w), 4]
        """
        cls_logits, ctr_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]

        cls_targets_all_level = []
        ctr_targets_all_level = []
        reg_targets_all_level = []

        assert len(self.strides) == len(cls_logits)

        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], ctr_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            ctr_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level, dim=1), torch.cat(ctr_targets_all_level, dim=1), torch.cat(
            reg_targets_all_level, dim=1)
