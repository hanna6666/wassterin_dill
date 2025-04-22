import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso

class SparseFD(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(SparseFD, self).__init__()
        self.student = student
        self.teacher = teacher.eval()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.lambda_sparse = cfg.DISTILLER.LOSS_WEIGHT
        self.layer_id = 2  # 使用 f2_pre 作为中间层
        self.dictionary = torch.load(cfg.DISTILLER.DICT_PATH).T  # shape [C, K]

    def get_extra_parameters(self):
        return 0  # 没有额外参数

    def forward(self, image, target, **kwargs):
        with torch.no_grad():
            _, t_feats = self.teacher(image)

        s_logits, s_feats = self.student(image)
        loss_cls = self.ce_loss(s_logits, target)

        t_feat = t_feats["preact_feats"][self.layer_id]
        s_feat = s_feats["preact_feats"][self.layer_id]
        loss_sparse = self.sparse_reconstruct_loss(s_feat, self.dictionary)

        loss = loss_cls + self.lambda_sparse * loss_sparse
        return {
            "logits_student": s_logits,
            "logits_teacher": None,
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_kd": loss_sparse,
        }

    def sparse_reconstruct_loss(self, F_S, D):
        B, C, H, W = F_S.shape
        loss = 0.0
        for i in range(B):
            f_s = F_S[i].mean(dim=(1, 2)).detach().cpu().numpy()
            lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=1000)
            lasso.fit(D.cpu().numpy().T, f_s)
            x_i = torch.tensor(lasso.coef_).to(F_S.device)
            f_hat = torch.matmul(D, x_i)
            loss += F.mse_loss(F_S[i].mean(dim=(1, 2)), f_hat)
        return loss / B