import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import sys
cudnn.benchmark = True
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset, cifar100, imagenet
from mdistiller.dataset.imagenet import *
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg
import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def genarate_class_features(loader, distiller, num_class=100, feature_type='logit'):
    class_feat = [[] for _ in range(num_class)]
    distiller.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm.tqdm(loader)):
            if len(data) == 3:
                image, target, index = data
            elif len(data) == 2:
                image, target = data
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # with torch.cuda.amp.autocast():
            logits, feats = distiller.forward_all(image=image)
            
            for i in range(logits.shape[0]):
                if feature_type == 'logit':
                    class_feat[target[i]].append(logits[i, ...].cpu())
                else:
                    class_feat[target[i]].append(feats['feats'][-1][i, ...].cpu())

        if feature_type == 'logit':
            min_samples = min([len(class_feat[i]) for i in range(num_class)])
            class_feat = torch.stack([torch.stack(class_feat[i][:min_samples], dim=0) for i in range(num_class)], dim=0)
        else:
            class_feat = torch.stack([torch.stack(class_feat[i], dim=0) for i in range(num_class)], dim=0)
            class_feat_reshaped = []
            for cls_idx, item in enumerate(class_feat):
                feats = item
                m,c,h,w= feats.shape
                feats = feats.view(m, c, -1)
                feats = feats.transpose(-1,-2)
                feats = feats.reshape(-1, c)
                class_feat_reshaped.append(feats)
            class_feat=torch.stack(class_feat_reshaped, dim=0)

    print(class_feat.shape)
    return class_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ResNet34")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar100", "imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=512)
    parser.add_argument("--save_name", type=str, default="features_train")
    parser.add_argument("--feature_type", type=str, default="logit")
    args = parser.parse_args()
    
    set_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    cfg.SOLVER.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        test_transform = get_imagenet_test_transform(mean, std)
        data_folder = imagenet.data_folder
        train_folder = os.path.join(data_folder, 'train')
        train_set = ImageNet(train_folder, transform=test_transform)
       
        train_loader = torch.utils.data.DataLoader(train_set,
            batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=seed_worker,
                    generator=g)
        # print(len(train_loader))
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            print("Load from")
            print(args.ckpt)
            model = imagenet_model_dict[args.model](pretrained=False, num_classes=1000)

            def on_load_checkpoint(state):
                key_list = list(state['model'].keys())
                for key in key_list:
                    if 'orig_mod.' in key:
                        deal_key = key.replace('_orig_mod.', '')
                        state['model'][deal_key] = state['model'][key]
                        del state['model'][key]
                return state

            path = os.path.join("imagenet_models", args.model, args.ckpt)
            state = torch.load(path)
            state = on_load_checkpoint(state)
            model.load_state_dict(state['model'])
        

    elif args.dataset == "cifar100":
        data_folder = cifar100.get_data_folder()
        transform = cifar100.get_cifar100_test_transform()
        train_set = cifar100.CIFAR100Instance(root=data_folder, 
                                              download=True, 
                                              train=True, 
                                              transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=cfg.SOLVER.BATCH_SIZE, 
                                                   shuffle=False, 
                                                   num_workers=cfg.DATASET.NUM_WORKERS)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=100)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])

    model = Vanilla(model)
    model = model.cuda()
    if hasattr(model.student, "fc"):
        num_class, representation_dim = model.student.fc.weight.shape
    if hasattr(model.student, "head"):
        num_class, representation_dim = model.student.head.weight.shape
    if hasattr(model.student, "classifier"):
        num_class, representation_dim = model.student.classifier.weight.shape
    if hasattr(model.student, "linear"):
        num_class, representation_dim = model.student.linear.weight.shape

    class_logit = genarate_class_features(loader=train_loader, 
                                          num_class=num_class, 
                                          distiller=model, 
                                          feature_type=args.feature_type)

    # import pdb; pdb.set_trace()
    save_path = os.path.join('wkd_cost_matrix', args.dataset, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(class_logit, os.path.join(save_path, args.save_name+".pth"))