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
    def __init__(self, student, teacher, cfg, max_buffer_size=3000,num_classes=100,warmup_steps=100, refresh_epoch=50,confidence_thresh=0.7,num_atoms=100, momentum=0.9, device='cuda'):
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
        self.fallback_layer = dict()

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

    # def update(self, feats, labels, logits,layer, epoch):
    #     """
    #     Update class-wise dictionaries with high-confidence teacher features.

    #     Args:
    #         feats: [B, C, H, W] feature maps from teacher model
    #         labels: [B] class labels
    #         logits: [B, num_classes] teacher logits
    #         epoch: int, current epoch number
    #     """
    #     C = feats.shape[1]
    #     if layer not in self.fallback_layer or self.fallback_layer[layer].shape[0] != C:
    #         print(f"[Init] fallback for layer {layer}: channel={C}")
    #         self.fallback_layer[layer] = torch.randn(C, self.num_atoms, device=self.device)
    #     if not hasattr(self, "class_dicts_layer"):
    #         self.class_dicts_layer = dict()
    #         self.class_buffers_layer = dict()
    #         self.initialized_classes_layer = dict()

    #     if layer not in self.class_dicts_layer:
    #         self.class_dicts_layer[layer] = {c: torch.zeros(C, self.num_atoms, device=self.device) for c in range(self.num_classes)}
    #         self.class_buffers_layer[layer] = defaultdict(list)
    #         self.initialized_classes_layer[layer] = [False] * self.num_classes

    #     step_key = f"step_{layer}"
    #     if not hasattr(self, step_key):
    #         setattr(self, step_key, 0)
    #     setattr(self, step_key, getattr(self, step_key) + 1)
    #     warmup = getattr(self, step_key) <= self.warmup_steps

    #     B = feats.shape[0]
    #     gap_feats = feats.mean(dim=(2, 3))  # [B, C]
    #     probs = F.softmax(logits, dim=1)
    #     confs, preds = probs.max(dim=1)

    #     for i in range(B):
    #         label_i = labels[i].item()
    #         pred_i = preds[i].item()
    #         conf_i = confs[i].item()
    #         feat_i = gap_feats[i].detach().cpu().numpy()

    #         if warmup or (conf_i >= self.conf_thresh and label_i == pred_i):
    #             self.class_buffers_layer[layer][label_i].append({'feat': feat_i, 'conf': conf_i})
    #             if len(self.class_buffers_layer[layer][label_i]) > self.max_buffer_size:
    #                 self.class_buffers_layer[layer][label_i].pop(0)

    #     for c in range(self.num_classes):
    #         buffer = self.class_buffers_layer[layer][c]
    #         if len(buffer) >= self.num_atoms:
    #             if not warmup:
    #                 sorted_buffer = sorted(buffer, key=lambda x: x['conf'], reverse=True)
    #                 buffer = sorted_buffer[:self.max_buffer_size]
    #                 self.class_buffers_layer[layer][c] = buffer
    #             buffer_np = np.stack([x['feat'] for x in buffer], axis=0)
    #             kmeans = KMeans(n_clusters=self.num_atoms, random_state=0).fit(buffer_np)
    #             new_dict = torch.tensor(kmeans.cluster_centers_.T, dtype=torch.float32, device=self.device)

    #             if not self.initialized_classes_layer[layer][c] or (epoch % self.refresh_epoch == 0):
    #                 self.class_dicts_layer[layer][c] = new_dict
    #                 self.initialized_classes_layer[layer][c] = True
    #             else:
    #                 self.class_dicts_layer[layer][c] = self.momentum * self.class_dicts_layer[layer][c] + (1 - self.momentum) * new_dict

    #             self.class_buffers_layer[layer][c].clear()

    #     # gather valid dicts
    #     unique_labels = torch.unique(labels)
    #     valid_dicts = [self.class_dicts_layer[layer][c.item()] for c in unique_labels if self.initialized_classes_layer[layer][c.item()]]

    #     if valid_dicts:
    #         D_batch = torch.cat(valid_dicts, dim=1)
    #         self.fallback_layer[layer] = D_batch
    #     else:
    #         D_batch = self.fallback_layer[layer]

    #     return D_batch

    # def update(self, batch_feats):
    #   """
    #   batch_feats: [B, C, H, W] from teacher
    #   """
    #   feats = batch_feats.mean(dim=(2, 3)).cpu().numpy()  # [B, C]
    #   kmeans = KMeans(n_clusters=min(self.num_atoms, len(feats)), random_state=0).fit(feats)
    #   D_batch = torch.tensor(kmeans.cluster_centers_.T).float().to(self.device)  # [C, K]

    #   if not self.initialized:
    #       self.dictionary = D_batch
    #       self.initialized = True
    #   else:
    #       self.dictionary = self.momentum * self.dictionary + (1 - self.momentum) * D_batch

    #   return self.dictionary
    def update(self, batch_feats, layer):
        """
        Args:
            batch_feats: [B, C, H, W] from teacher
            layer: int or str, used as dictionary key
        Returns:
            D_batch: Tensor of shape [C, K]
        """
        B, C, H, W = batch_feats.shape
        feats = batch_feats.mean(dim=(2, 3)).cpu().numpy()  # [B, C]

        # 去除重复样本
        feats = np.unique(feats, axis=0)
        n_clusters = min(self.num_atoms, len(feats))

        # KMeans 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feats)
        centers = torch.tensor(kmeans.cluster_centers_.T, dtype=torch.float32, device=self.device)  # [C, K]

        # Padding 以保持 [C, num_atoms] 输出
        if n_clusters < self.num_atoms:
            pad = torch.randn(C, self.num_atoms - n_clusters, device=self.device)
            D_batch = torch.cat([centers, pad], dim=1)
        else:
            D_batch = centers

        # 初始化 dictionary_layer 字典
        if not hasattr(self, "dictionary_layer"):
            self.dictionary_layer = dict()

        if layer not in self.dictionary_layer:
            self.dictionary_layer[layer] = D_batch
        else:
            self.dictionary_layer[layer] = self.momentum * self.dictionary_layer[layer] + (1 - self.momentum) * D_batch

        return self.dictionary_layer[layer]

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
            t_feat_3 = feats_teacher["preact_feats"][3]
            s_feat_3 = feats_student["preact_feats"][3]

            t_feat_2 = feats_teacher["preact_feats"][2]
            s_feat_2 = feats_student["preact_feats"][2]

            t_feat_1 = feats_teacher["preact_feats"][1]
            s_feat_1 = feats_student["preact_feats"][1]

            t_feat_0 = feats_teacher["preact_feats"][0]
            s_feat_0 = feats_student["preact_feats"][0]

        logits_student = logits_student.to(torch.float32)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 更新字典 + 注意力对齐 loss
        # D_momentum_3 = self.update(t_feat_3, target, logits_teacher,3, kwargs['epoch'])
        # loss_attn_3 = self.attn_loss_weight * self.attention_align_loss(t_feat_3.float(), s_feat_3.float(), D_momentum_3)

        # D_momentum_2 = self.update(t_feat_2, target, logits_teacher,2, kwargs['epoch'])
        # loss_attn_2 = self.attn_loss_weight * self.attention_align_loss(t_feat_2.float(), s_feat_2.float(), D_momentum_2)

        # D_momentum_1 = self.update(t_feat_1, target, logits_teacher,1, kwargs['epoch'])
        # loss_attn_1 = self.attn_loss_weight * self.attention_align_loss(t_feat_1.float(), s_feat_1.float(), D_momentum_1)

        # D_momentum_0 = self.update(t_feat_0, target, logits_teacher,0, kwargs['epoch'])
        # loss_attn_0 = self.attn_loss_weight * self.attention_align_loss(t_feat_0.float(), s_feat_0.float(), D_momentum_0)

        D_momentum_3 = self.update(t_feat_3,3)
        loss_attn_3 = self.attn_loss_weight * self.attention_align_loss(t_feat_3.float(), s_feat_3.float(), D_momentum_3)

        D_momentum_2 = self.update(t_feat_2,2)
        loss_attn_2 = self.attn_loss_weight * self.attention_align_loss(t_feat_2.float(), s_feat_2.float(), D_momentum_2)

        D_momentum_1 = self.update(t_feat_1,1)
        loss_attn_1 = self.attn_loss_weight * self.attention_align_loss(t_feat_1.float(), s_feat_1.float(), D_momentum_1)

        D_momentum_0 = self.update(t_feat_0,0)
        loss_attn_0 = self.attn_loss_weight * self.attention_align_loss(t_feat_0.float(), s_feat_0.float(), D_momentum_0)
        

        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
          # cosine decay
            self.wkd_logit_loss_weight_1 = 0.5*self.wkd_logit_loss_weight*(1+math.cos((kwargs['epoch']-decay_start_epoch)/(self.cfg.SOLVER.EPOCHS-decay_start_epoch)*math.pi))
          
        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight
        if self.enable_wkdl:
            logits_teacher = logits_teacher.to(torch.float32)
            loss_wkd_logit = wkd_logit_loss_with_speration(logits_student, logits_teacher, target, self.temperature, self.wkd_logit_loss_weight_1, self.dist, self.sinkhorn_lambda, self.sinkhorn_iter)

        total_loss = loss_attn_3 + loss_attn_2 +  loss_attn_1 + loss_attn_0 + loss_wkd_logit

        loss_attn = loss_attn_3 + loss_attn_2 +  loss_attn_1 + loss_attn_0

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_attn,
            "loss_wkd":loss_wkd_logit,
            # "loss_kd": total_loss,
        }

        return logits_student, losses_dict

    def get(self):
        return self.dictionary