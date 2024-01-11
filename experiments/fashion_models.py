import timm
from enum import StrEnum
from typing import Optional, List
from fnmatch import fnmatch
import torch
import torch.nn as nn


class ModelName(StrEnum):
    convnext_atto = "convnext_atto"
    convnext_maxplus_atto = "convnext_maxplus_atto"
    convnext_minplus_atto = "convnext_minplus_atto"
    convnext_log_m1_atto = "convnext_log_m1_atto"
    convnext_log_p1_atto = "convnext_log_p1_atto"


def list_models(filter: str = "*") -> List[str]:
    """
    Gives a list of the names of the available models, subject to `filter`.
    """
    return [str(x) for x in ModelName if fnmatch(str(x), filter)]


def create_model(name: ModelName):
    if name == ModelName.convnext_atto:
        return timm.create_model(
            "convnext_atto", in_chans=1, patch_size=2, num_classes=10
        )
    else:
        raise NotImplementedError()


def create_optimizers_and_schedulers(name: ModelName, batchsize:int, epochs: int):
    pass


def resetmodel(model: nn.Module) -> None:
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(fn=weight_reset)