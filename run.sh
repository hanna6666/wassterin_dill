# single gpu
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./configs/imagenet/r34_r18/wkd_l.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./configs/imagenet/r34_r18/wkd_f.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./configs/imagenet/r34_r18/wkd_lf.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./configs/imagenet/r50_mv1/wkd_l.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./configs/imagenet/r50_mv1/wkd_f.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg ./configs/imagenet/r50_mv1/wkd_lf.yaml

# multi-gpu
# CUDA_VISIBLE_DEVICES=0,1 torchrun --master-port 25900 --nproc-per-node=2 tools/train_with_ddp.py --cfg ./configs/imagenet/r50_mv1/wkd_l.yaml 
# CUDA_VISIBLE_DEVICES=2,3 torchrun --master-port 25300 --nproc-per-node=2 tools/train_with_ddp.py --cfg ./configs/imagenet/r50_mv1/wkd_f.yaml 