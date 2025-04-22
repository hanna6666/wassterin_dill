import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    if hasattr(val_loader, 'loader'):
        num_iter = len(val_loader.loader)
    else:
        num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image,target) in enumerate(val_loader):
            # image, target = data[0]["data"], data[0]["label"].squeeze(-1).long()
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # with torch.cuda.amp.autocast():
            # print(image.shape, target.shape)
            output = distiller(image=image)
            # print(output.shape, target.shape)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg

def validate_teacher(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    if hasattr(val_loader, 'loader'):
        num_iter = len(val_loader.loader)
    else:
        num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))
    distiller = distiller.cuda()
    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output, _ = distiller(image)
            # print(output.shape)
            # exit()
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg

def adjust_warmup(iter, warm_iter, cfg, optimizer):
    if iter < warm_iter:
        new_lr = cfg.SOLVER.LR * (iter + 1) / warm_iter
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return optimizer.param_groups[0]["lr"]
    
def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR

def adjust_warmup(iter, warm_iter, cfg, optimizer):
    if iter < warm_iter:
        new_lr = cfg.SOLVER.LR * (iter + 1) / warm_iter
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return optimizer.param_groups[0]["lr"]

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    print(pth)
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)