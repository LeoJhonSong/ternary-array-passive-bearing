from abc import ABC, abstractmethod
from yacs.config import CfgNode as CN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class Estimator(ABC):
    def __init__(self, sig_cfg: CN):
        self.sig_cfg = sig_cfg
        self.c = sig_cfg.c
        self.f = sig_cfg.f
        self.fs = sig_cfg.fs

    @abstractmethod
    def __call__(self, x: 'np.ndarray'):
        pass
