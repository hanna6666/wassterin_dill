import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdistiller.dataset.cifar100 import get_cifar100_train_transform, CIFAR100Instance, get_data_folder
from mdistiller.models import cifar_model_dict
from mdistiller.engine.cfg import CFG as cfg
import numpy as np
import os

def extract_features(model, dataloader, layer_id=2, max_batches=500):
    model.eval()
    features = []

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(dataloader)):
            if i >= max_batches:
                break
            x = x.cuda()
            _, feat_dict = model(x)
            f = feat_dict["preact_feats"][layer_id]  # f2_pre
            f = f.mean(dim=(2, 3))  # Global average pooling [B, C]
            features.append(f.cpu())

    return torch.cat(features, dim=0)  # shape [N, C]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--cfg", type=str, default="")
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 设置
    teacher_name = "resnet34x4"
    num_classes = 100
    save_path = f"dicts/{teacher_name}_f2pre_dict.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 数据
    data_folder = get_data_folder()
    transform = get_cifar100_train_transform()
    dataset = CIFAR100Instance(root=data_folder, train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 教师模型
    net, path = cifar_model_dict[cfg.DISTILLER.TEACHER]
    teacher = net(num_classes=num_classes).cuda()
    teacher.load_state_dict(torch.load(path,weights_only=False)["model"])

    # 提取特征
    feats = extract_features(teacher, loader, layer_id=2)  # shape [N, C]

    print("Extracted features:", feats.shape)  # 例如 [64000, 64]

    # 保存
    torch.save(feats, save_path)
    print(f"Saved to {save_path}")
