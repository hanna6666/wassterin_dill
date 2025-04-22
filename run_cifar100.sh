# single gpu


CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/wkd_l/res32x4_res8x4.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/wkd_f/res32x4_res8x4.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/wkd_lf/res32x4_res8x4.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/wkd_l/wrn40_2_shuv1.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/wkd_f/wrn40_2_shuv1.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/wkd_lf/wrn40_2_shuv1.yaml

