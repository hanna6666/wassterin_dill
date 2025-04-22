import os
import torch.backends.cudnn as cudnn
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import *
from mdistiller.engine.utils import load_checkpoint, validate, validate_teacher
from mdistiller.engine.cfg import CFG as cfg

def on_load_checkpoint(state):
                key_list = list(state['model'].keys())
                for key in key_list:
                    if 'orig_mod.' in key:
                        deal_key = key.replace('_orig_mod.', '')
                        state['model'][deal_key] = state['model'][key]
                        del state['model'][key]
                return state
            

def on_load_nkd(state):
    key_list = list(state['state_dict'].keys())
    for key in key_list:
        if 'backbone.' in key:
            deal_key = key.replace('backbone.', '')
            state['state_dict'][deal_key] = state['state_dict'][key]
            del state['state_dict'][key]
        if 'head.' in key:
            deal_key = key.replace('head.', '')
            state['state_dict'][deal_key] = state['state_dict'][key]
            del state['state_dict'][key]
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ResNet18")
    parser.add_argument("-c", "--ckpt", type=str, default="/home/mdistiller/imagenet_models/ResNet18/Res18_nkd_71.96.pth")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar100", "imagenet","imagenetv2","imagenet-sketch"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=512)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    
    if "imagenet" in args.dataset:
        if args.dataset == "imagenet":
            val_loader = get_imagenet_val_loader(args.batch_size)
        if args.dataset == "imagenetv2":
            val_loader = get_imagenet_v2_loader(args.batch_size)
        if args.dataset == "imagenet-sketch":
            val_loader = get_imagenet_sketch_loader(args.batch_size)

        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
            print("load pretrain")
        else:    
            model = imagenet_model_dict[args.model](pretrained=False)
            w = torch.load(args.ckpt)
            if 'nkd' in args.ckpt:
                w = on_load_nkd(w)
                model.load_state_dict(w["state_dict"])
            else:
                w = on_load_checkpoint(w)
                model.load_state_dict(w["model"])
            print("load ckpt")
            model = model# .cuda()
            model.eval()


    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])
        
    model = Vanilla(model)
    model = model.cuda()
    model.eval()
    # model = torch.nn.DataParallel(model)
    test_acc, test_acc_top5, test_loss = validate(val_loader, model)
    
