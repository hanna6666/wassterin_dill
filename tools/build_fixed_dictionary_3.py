import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import sys

# === 路径注册 ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdistiller.dataset.cifar100 import get_cifar100_train_transform, CIFAR100Instance, get_data_folder
from mdistiller.models import cifar_model_dict
from mdistiller.engine.cfg import CFG as cfg


def extract_features(model, dataloader, layer_id=2, max_batches=500):
    model.eval()
    features, all_labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader)):
            if i >= max_batches:
                break
            x = x.cuda()
            _, feat_dict = model(x)
            f = feat_dict["preact_feats"][layer_id].mean(dim=(2, 3))  # [B, C]
            features.append(f.cpu())
            all_labels.append(y)
    return torch.cat(features), torch.cat(all_labels)


def build_fixed_dict(feats, labels, num_classes=100, num_atoms_per_class=3):
    per_class_feats = defaultdict(list)
    for f, label in zip(feats, labels):
        per_class_feats[label.item()].append(f.numpy())

    dict_atoms = []
    for c in range(num_classes):
        feats_c = np.stack(per_class_feats[c], axis=0)
        if len(feats_c) < num_atoms_per_class:
            pad = feats_c[np.random.choice(len(feats_c), num_atoms_per_class - len(feats_c))]
            feats_c = np.concatenate([feats_c, pad], axis=0)
        kmeans = KMeans(n_clusters=num_atoms_per_class, random_state=0).fit(feats_c)
        centers = kmeans.cluster_centers_.T  # [C, 3]
        dict_atoms.append(centers)
    dictionary = np.concatenate(dict_atoms, axis=1)  # [C, 3 * num_classes]
    return torch.tensor(dictionary, dtype=torch.float32)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Build fixed dictionary for attention distillation")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    teacher_name = cfg.DISTILLER.TEACHER
    num_classes = 100
    layer_id = 3
    num_atoms_per_class = 3
    max_batches = 500

    save_path = f"/kaggle/working/wassterin_dill/dicts/{teacher_name}_fixed_dict_3.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    transform = get_cifar100_train_transform()
    dataset = CIFAR100Instance(root=get_data_folder(), train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    net_fn, path = cifar_model_dict[teacher_name]
    teacher = net_fn(num_classes=num_classes).cuda()
    teacher.load_state_dict(torch.load(path, weights_only=False)["model"])

    feats, labels = extract_features(teacher, loader, layer_id=layer_id, max_batches=max_batches)
    fixed_dict = build_fixed_dict(feats, labels, num_classes, num_atoms_per_class)

    torch.save(fixed_dict, save_path)
    print(f"Saved fixed dictionary to {save_path} with shape {fixed_dict.shape}")
