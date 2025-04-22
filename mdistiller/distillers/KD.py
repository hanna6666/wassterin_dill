import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.ir_loss_weight = cfg.KD.LOSS.IR_WEIGHT


    def forward_train(self, image, target, **kwargs):
        with torch.cuda.amp.autocast():
            logits_student, _ = self.student(image)
            with torch.no_grad():
                logits_teacher, _ = self.teacher(image)

        logits_student = logits_student.to(torch.float32)
        logits_teacher = logits_teacher.to(torch.float32)

        ws = F.normalize(self.student.fc.weight, dim=-1)
        ir_s = ws @ ws.T
        wt = F.normalize(self.teacher.fc.weight, dim=-1)
        ir_t = wt @ wt.T

        loss_ir = self.ir_loss_weight * F.mse_loss(ir_s, ir_t, reduction='sum')
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_ir": loss_ir
        }
        # print(losses_dict)
        return logits_student, losses_dict
    
    def forward_train_clip(self, image, target, text_features=None, **kwargs):
        with torch.cuda.amp.autocast():
            logits_student, _ = self.student(image)

            # save_path = '/media/ljm/ImageNet200_CLIP_ViT-B32_pre-extracted-logits/epoch_{}/iter_{}.pth'.format(kwargs['epoch'], kwargs['iter'])
            # logits_teacher = torch.load(save_path).cuda()

            with torch.no_grad():
                image_features = self.teacher.encode_image(image)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # logit_scale = self.teacher.logit_scale.exp()
            # logits_per_image = logit_scale * image_features @ text_features
            # logits_per_image = kwargs['linear_clf'](image_features.float())
            # logits_teacher = logits_per_image

        # if kwargs['iter'] % 200 == 0:
        #     print(logits_teacher - logits_per_image)

        # ===== extract teacher pred logits, add 20230613 =====
        # save_path = '/media/ljm/ImageNet200_CLIP_ViT-B32_finetuned_pre-extracted-logits/epoch_{}'.format(kwargs['epoch'])
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(logits_teacher.cpu(), os.path.join(save_path, 'iter_{}.pth'.format(kwargs['iter'])))
        # =======================================

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }

        return logits_student, losses_dict

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.student(kwargs["image"])[0]
