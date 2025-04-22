from ._base import Vanilla
from .KD import KD
from .FitNet import FitNet
from .CRD import CRD
from .DKD import DKD
from .WKD import WKD
from .sparsefd import SparseFD
from .momentum_dictionary import MomentumDictionary

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "FITNET": FitNet,
    "CRD": CRD,
    "DKD": DKD,
    "WKD": WKD,
    "SPARSEFD": SparseFD,
    "MomentumDictionary": MomentumDictionary 
}
