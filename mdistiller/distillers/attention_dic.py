import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
import math
from ._base import Distiller
from ._common import *
import numpy as np
from collections import defaultdict

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

def wkd_logit_loss_with_speration(logits_student, logits_teacher, gt_label, temperature, gamma, cost_matrix=None, sinkhorn_lambda=0.05, sinkhorn_iter=10):
        
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

    # N*class
    N, c = logits_student.shape
    s_i = F.log_softmax(logits_student, dim=1)
    t_i = F.softmax(logits_teacher, dim=1)
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()
    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_student = logits_student[mask].reshape(N, -1)
    logits_teacher = logits_teacher[mask].reshape(N, -1)
    
    cost_matrix = cost_matrix.repeat(N, 1, 1)
    gd_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    cost_matrix = cost_matrix[gd_mask].reshape(N, c-1, c-1)
        
    # N*class
    loss_wkd = wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix, sinkhorn_lambda, sinkhorn_iter)

    return loss_t + gamma * loss_wkd


class AttentionMapDistiller(Distiller):
    def __init__(self, student, teacher, cfg, max_buffer_size=100,num_classes=100,warmup_steps=10, refresh_epoch=50,confidence_thresh=0.8,num_atoms=100, momentum=0.9, device='cuda'):
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
        self.wkd_logit_loss_weight = cfg.WKD.LOSS.WKD_LOGIT_WEIGHT
        self.loss_cosine_decay_epoch = cfg.WKD.LOSS.COSINE_DECAY_EPOCH
        self.conf_thresh = confidence_thresh
        self.step = 0
        self.refresh_epoch = refresh_epoch
        self.warmup_steps = warmup_steps
        self.num_classes = num_classes
        self.max_buffer_size = max_buffer_size

        self.enable_wkdl = cfg.WKD.LOSS.WKD_LOGIT_WEIGHT > 0
        if self.enable_wkdl:
            self.temperature = cfg.WKD.TEMPERATURE
            self.sinkhorn_lambda = cfg.WKD.SINKHORN.LAMBDA
            self.sinkhorn_iter = cfg.WKD.SINKHORN.ITER

            if cfg.WKD.COST_MATRIX == "fc":
              print("Using fc weight of teacher model as category prototype")
              self.prototype = self.teacher.fc.weight
              # caluate cosine similarity
              proto_normed = F.normalize(self.prototype, p=2, dim=-1)
              cosine_sim = proto_normed.matmul(proto_normed.transpose(-1, -2))
              self.dist = 1 - cosine_sim
            else:
              print("Using "+cfg.WKD.COST_MATRIX+" as cost matrix")
              path_gd = cfg.WKD.COST_MATRIX_PATH
              self.dist = torch.load(path_gd).cuda().detach()
              
            if cfg.WKD.COST_MATRIX_SHARPEN != 0:
              print("Sharpen ", cfg.WKD.COST_MATRIX_SHARPEN)
              sim = torch.exp(-cfg.WKD.COST_MATRIX_SHARPEN*self.dist)
              self.dist = 1 - sim

    def get_extra_parameters(self):
        return 0

    # def update(self, batch_feats):
    #     """
    #     batch_feats: [B, C, H, W] from teacher
    #     """
    #     feats = batch_feats.mean(dim=(2, 3)).cpu().numpy()  # [B, C]
    #     kmeans = KMeans(n_clusters=min(self.num_atoms, len(feats)), random_state=0).fit(feats)
    #     D_batch = torch.tensor(kmeans.cluster_centers_.T).float().to(self.device)  # [C, K]

    #     if not self.initialized:
    #         self.dictionary = D_batch
    #         self.initialized = True
    #     else:
    #         self.dictionary = self.momentum * self.dictionary + (1 - self.momentum) * D_batch

    #     return self.dictionary

    def update(self, feats, labels, logits, epoch):
        """
        Update class-wise dictionaries with high-confidence teacher features.

        Args:
            feats: [B, C, H, W] feature maps from teacher model
            labels: [B] class labels
            logits: [B, num_classes] teacher logits
            epoch: int, current epoch number
        """
        if not hasattr(self, "fallback") or self.fallback is None:
            self.fallback = torch.randn(feats.shape[1], self.num_atoms, device=self.device)
        if not hasattr(self, "class_dicts"):
            self.class_dicts = {c: torch.zeros(feats.shape[1], self.num_atoms, device=self.device) for c in range(self.num_classes)}
            self.initialized_classes = [False] * self.num_classes
            self.class_buffers = defaultdict(list)

        self.step += 1
        warmup = self.step <= self.warmup_steps
        B, C, H, W = feats.shape
        gap_feats = feats.mean(dim=(2, 3))  # [B, C]
        probs = F.softmax(logits, dim=1)  # [B, num_classes]
        confs, preds = probs.max(dim=1)   # [B]

        for i in range(B):
            label_i = labels[i].item()
            pred_i = preds[i].item()
            conf_i = confs[i].item()
            feat_i = gap_feats[i].detach().cpu().numpy()

            # 置信度高且预测正确的样本
            if warmup or conf_i >= self.conf_thresh and label_i == pred_i:
                self.class_buffers[label_i].append({'feat': feat_i, 'conf': conf_i})
            # 限制 buffer 大小
                if len(self.class_buffers[label_i]) > self.max_buffer_size:
                    self.class_buffers[label_i].pop(0)

        for c in range(self.num_classes):
            buffer = self.class_buffers[c]
            if len(buffer) >= self.num_atoms:
                if not warmup:
                    sorted_buffer = sorted(buffer, key=lambda x: x['conf'], reverse=True)
                    buffer = sorted_buffer[:self.max_buffer_size]
                    self.class_buffers[c] = buffer
                # 多样性通过 KMeans 聚类自动获得
                buffer_np = np.stack(buffer, axis=0)
                kmeans = KMeans(n_clusters=self.num_atoms, random_state=0).fit(buffer_np)
                new_dict = torch.tensor(kmeans.cluster_centers_.T, dtype=torch.float32, device=self.device)  # [C, K]

                if not self.initialized[c] or (epoch % self.refresh_epoch == 0):
                    self.class_dicts[c] = new_dict
                    self.initialized[c] = True
                else:
                    self.class_dicts[c] = self.momentum * self.class_dicts[c] + (1 - self.momentum) * new_dict

                self.class_buffers[c].clear()

        unique_labels = torch.unique(labels)
        valid_dicts = [self.class_dicts[c.item()] for c in unique_labels if self.initialized_classes[c.item()]]

        if valid_dicts:
            D_batch = torch.cat(valid_dicts, dim=1)  # [C, K × n]
            self.fallback = D_batch
        else:
            D_batch = self.fallback

        return D_batch

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
        # D_momentum = self.update(t_feat)
        D_momentum = self.update(t_feat, target, logits_teacher, kwargs['epoch'])
        loss_attn = self.attn_loss_weight * self.attention_align_loss(t_feat.float(), s_feat.float(), D_momentum)

        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
          # cosine decay
            self.wkd_logit_loss_weight_1 = 0.5*self.wkd_logit_loss_weight*(1+math.cos((kwargs['epoch']-decay_start_epoch)/(self.cfg.SOLVER.EPOCHS-decay_start_epoch)*math.pi))
          
        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight
        if self.enable_wkdl:
            logits_teacher = logits_teacher.to(torch.float32)
            loss_wkd_logit = wkd_logit_loss_with_speration(logits_student, logits_teacher, target, self.temperature, self.wkd_logit_loss_weight_1, self.dist, self.sinkhorn_lambda, self.sinkhorn_iter)

        total_loss = loss_attn + loss_wkd_logit

        losses_dict = {
            "loss_ce": loss_ce,
            # "loss_kd": loss_attn,
            "loss_kd": total_loss,
        }

        return logits_student, losses_dict

    def get(self):
        return self.dictionary