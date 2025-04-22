from .trainer import BaseTrainer, CRDTrainer
from .trainer_with_ddp import BaseTrainer as BaseTrainerDDP, CRDTrainer as CRDTrainerDDP

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
}

trainer_dict_ddp = {
    "base": BaseTrainerDDP,
    "crd": CRDTrainerDDP,
}

