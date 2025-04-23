import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
import math
from ._base import Distiller
from ._common import *

class AttentionMapDistiller(Distiller):
    def __init__(self, student, teacher, cfg, num_atoms=100, momentum=0.9, device='cuda'):
        super(AttentionMapDistiller, self).__init__(student, teacher)
        self.cfg = cfg
        self.num_atoms = num_atoms
        self.momentum = momentum
        self.device = device
        self.initialized = False
        self.dictionary = None  # [C, K]
        self.student = self.student.to(self.device)
        self.teacher = self.teacher.to(self.device).eval()
        self.ce_loss_weight = cfg.WKD.LOSS.CE_WEIGHT
        self.attn_loss_weight = cfg.WKD.LOSS.FEAT_WEIGHT  # 用于 attention loss 的权重

    def get_extra_parameters(self):
        return 0

    def update(self, batch_feats):
        """
        batch_feats: [B, C, H, W] from teacher
        """
        feats = batch_feats.mean(dim=(2, 3)).cpu().numpy()  # [B, C]
        kmeans = KMeans(n_clusters=min(self.num_atoms, len(feats)), random_state=0).fit(feats)
        D_batch = torch.tensor(kmeans.cluster_centers_.T).float().to(self.device)  # [C, K]

        if not self.initialized:
            self.dictionary = D_batch
            self.initialized = True
        else:
            self.dictionary = self.momentum * self.dictionary + (1 - self.momentum) * D_batch

        return self.dictionary

    def attention_align_loss(self, F_T, F_S, D):
        """
        F_T: [B, C, H, W] teacher features
        F_S: [B, C, H, W] student features
        D: [C, K] dictionary
        """
        B, C, H, W = F_T.shape
        F_T = F_T.mean(dim=(2, 3))  # [B, C]
        F_S = F_S.mean(dim=(2, 3))  # [B, C]

        scale = math.sqrt(C)

        # D.T: [K, C], F.T: [C, B] → [K, B]
        A_T = torch.softmax(torch.matmul(D.T, F_T.T) / scale, dim=0)  # [K, B]
        A_S = torch.softmax(torch.matmul(D.T, F_S.T) / scale, dim=0)  # [K, B]

        loss_attn = F.mse_loss(A_S, A_T.detach())  # detach teacher
        return loss_attn

    def forward_train(self, image, target, **kwargs):
        with torch.cuda.amp.autocast():
            logits_student, feats_student = self.student(image)
            with torch.no_grad():
                logits_teacher, feats_teacher = self.teacher(image)
            t_feat = feats_teacher["preact_feats"][3]
            s_feat = feats_student["preact_feats"][3]

        logits_student = logits_student.to(torch.float32)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 更新字典 + 注意力对齐 loss
        D_momentum = self.update(t_feat)
        loss_attn = self.attn_loss_weight * self.attention_align_loss(t_feat.float(), s_feat.float(), D_momentum)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_attn,
        }

        return logits_student, losses_dict

    def get(self):
        return self.dictionary