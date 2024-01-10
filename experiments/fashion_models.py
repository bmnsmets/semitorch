import timm
from enum import StrEnum
from typing import Optional, List
from fnmatch import fnmatch


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


def create_config(name: ModelName, batchsize:int, epochs: int):
    pass
