import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import sys
cudnn.benchmark = True
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset.imagenet import *
from mdistiller.dataset.cifar100 import *
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict_ddp

import numpy as np
import random
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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


def get_imagenet_dataloaders(batch_size, val_batch_size, num_workers, worker_init_fn, generator,input_size=224,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std, input_size)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn, generator=generator, sampler=train_sampler)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data


def get_cifar100_dataloaders(batch_size, val_batch_size, num_workers, worker_init_fn, generator,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(
        root=data_folder, download=True, train=True, transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb
            os.makedirs('/media/ljm/mdistiller0414/', exist_ok=True)
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags, dir='/media/ljm/mdistiller/')
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    if dist.get_rank() == 0:
        show_cfg(cfg)
    set_seed(rank+1)
    g = torch.Generator()
    g.manual_seed(rank+1)


    # init dataloader & models
    if cfg.DATASET.TYPE == "imagenet":
        train_loader, val_loader, num_data = get_imagenet_dataloaders(
                    batch_size=cfg.SOLVER.BATCH_SIZE,
                    val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                    num_workers=cfg.DATASET.NUM_WORKERS,
                    worker_init_fn=seed_worker,
                    generator=g,
                    input_size=cfg.DATASET.INPUT_SIZE,
                )
    else:
        train_loader, val_loader, num_data = get_cifar100_dataloaders(
                    batch_size=cfg.SOLVER.BATCH_SIZE,
                    val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                    num_workers=cfg.DATASET.NUM_WORKERS,
                    worker_init_fn=seed_worker,
                    generator=g,
                ) 
    num_classes = cfg.DATASET.NUM_CLASSES

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False, num_classes=num_classes)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False, num_classes=num_classes)
            
            # model_teacher = torch.compile(model_teacher, disable=True)
            # model_student = torch.compile(model_student, mode='reduce-overhead')


        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = distiller.cuda()
    distiller = distiller.to(local_rank)
    distiller = DDP(distiller, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    # cfg.DISTILLER.TYPE, distiller.get_extra_parameters()
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict_ddp[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg, local_rank=local_rank
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)