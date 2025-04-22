import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
from ._base import Distiller
from ._common import *
from sklearn.linear_model import Lasso

class MomentumDictionary(Distiller):
    def __init__(self, student, teacher, cfg, num_atoms=100, momentum=0.9, device='cuda'):
      super(MomentumDictionary, self).__init__(student, teacher)
      self.cfg = cfg
      self.num_atoms = num_atoms
      self.momentum = momentum
      self.device = device
      self.initialized = False
      self.dictionary = None  # [C, K]
      self.enable_wkdf = False
      self.student = self.student.to(self.device)
      self.teacher = self.teacher.to(self.device).eval()
      self.ce_loss_weight = cfg.WKD.LOSS.CE_WEIGHT
      self.fg_weight = cfg.WKD.LOSS.FEAT_WEIGHT

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

    def sparse_reconstruct_loss(self, F_S, D):
      B, C, H, W = F_S.shape
      loss = 0.0
      D_np = D.cpu().numpy()
      for i in range(B):
          feat = F_S[i].mean(dim=(1, 2)).detach().cpu().numpy()  # GAP → [C]
          lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=1000)
          lasso.fit(D_np, feat)
          x_i = torch.tensor(lasso.coef_, dtype=torch.float32).to(F_S.device)
          f_hat = torch.matmul(D, x_i)
          loss += F.mse_loss(F_S[i].mean(dim=(1, 2)), f_hat)
      return loss / B

    def forward_train(self, image, target, **kwargs):
      with torch.cuda.amp.autocast():
          logits_student, feats_student = self.student(image)
          with torch.no_grad():
              logits_teacher, feats_teacher = self.teacher(image)
          t_feat = feats_teacher["preact_feats"][4]
          s_feat = feats_student["preact_feats"][4]

      logits_student = logits_student.to(torch.float32)
      loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

      t_feat = t_feat.float()
      s_feat = s_feat.float()
      # s_feat_nor = F.normalize(s_feat, dim=1)

      D_momentum = self.update(t_feat)
      # D_nor = F.normalize(D_momentum, dim=0)
      loss_sparse = self.fg_weight * self.sparse_reconstruct_loss(s_feat, D_momentum)

      losses_dict = {
          "loss_ce": loss_ce,
          "loss_kd": loss_sparse,
      }

      return logits_student, losses_dict

    def get(self):
      return self.dictionary