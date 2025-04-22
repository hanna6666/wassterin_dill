import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes


class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitNet, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        # self.conv_reg = nn.Sequential(
        #     nn.Conv2d(in_channels=feat_s_shapes[-1][1], out_channels=feat_t_shapes[-1][1], kernel_size=1, bias=False),
        #     nn.BatchNorm2d(feat_t_shapes[-1][1]),
        #     nn.ReLU(inplace=True)
        # )                                

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        # f_s = self.conv_reg(feature_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1))
        # f_t = feature_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        # loss_feat = self.feat_loss_weight * F.mse_loss(f_s, f_t)
        # loss_feat = self.feat_loss_weight * (0.5 * (f_s - f_t)**2 - 1)
        # import pdb; pdb.set_trace()
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_wsd": loss_feat,
        }
        # import pdb; pdb.set_trace()
        return logits_student, losses_dict
