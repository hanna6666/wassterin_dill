import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
import math
from ._base import Distiller
from ._common import *

def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1 / dim * torch.ones_like(w1, device=w1.device, dtype=w1.dtype)
    K = torch.exp(-cost / reg)
    Kt = K.transpose(2, 1)
    for _ in range(max_iter):
        v = w2 / (torch.bmm(Kt, u) + 1e-8)
        u = w1 / (torch.bmm(K, v) + 1e-8)
    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow

def wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix, sinkhorn_lambda=0.05, sinkhorn_iter=30):
    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)
    cost_matrix = F.relu(cost_matrix) + 1e-8
    cost_matrix = cost_matrix.to(pred_student.device)
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)
    ws_distance = (flow * cost_matrix).sum(-1).sum(-1).mean()
    return ws_distance

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

        self.enable_wkdl = cfg.WKD.LOSS.WKD_LOGIT_WEIGHT > 0
        if self.enable_wkdl:
            self.wkd_logit_loss_weight = cfg.WKD.LOSS.WKD_LOGIT_WEIGHT
            self.temperature = cfg.WKD.TEMPERATURE
            self.sinkhorn_lambda = cfg.WKD.SINKHORN.LAMBDA
            self.sinkhorn_iter = cfg.WKD.SINKHORN.ITER

            if cfg.WKD.COST_MATRIX == "fc":
                print("Using fc weight of teacher model as category prototype")
                proto = self.teacher.fc.weight
                sim = F.normalize(proto, p=2, dim=-1) @ F.normalize(proto, p=2, dim=-1).T
                self.cost_matrix = 1 - sim
            else:
                print("Using external cost matrix")
                self.cost_matrix = torch.load(cfg.WKD.COST_MATRIX_PATH).to(self.device).detach()

            if cfg.WKD.COST_MATRIX_SHARPEN != 0:
                print("Sharpening cost matrix")
                sim = torch.exp(-cfg.WKD.COST_MATRIX_SHARPEN * self.cost_matrix)
                self.cost_matrix = 1 - sim

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

        if self.enable_wkdl:
            loss_logit = self.wkd_logit_loss_weight * wkd_logit_loss(
                logits_student, logits_teacher, self.temperature, self.cost_matrix,
                self.sinkhorn_lambda, self.sinkhorn_iter)
        else:
            loss_logit = torch.tensor(0.0, device=self.device)

        total_loss = loss_attn + loss_logit

        losses_dict = {
            "loss_ce": loss_ce,
            # "loss_kd": loss_attn,
            "loss_kd": total_loss,
        }

        return logits_student, losses_dict

    def get(self):
        return self.dictionary