import timm
from enum import StrEnum
from typing import Optional, List
from fnmatch import fnmatch
import torch
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
from types import SimpleNamespace


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
        model = timm.create_model(
            "convnext_atto", in_chans=1, patch_size=1, num_classes=10
        )
    else:
        raise NotImplementedError()

    return model


def create_optimizers_and_schedulers(
    model: nn.Module,
    cfg: SimpleNamespace,
):
    if cfg.modelname == ModelName.convnext_atto:
        cfg.lr = 4e-3
        cfg.weight_decay = 5e-3
        cfg.optimizer = "AdamW"
        cfg.schedule = "1-cycle cosine"
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        schd = CosineLRScheduler(
            opt,
            t_initial=cfg.epochs,
            warmup_t=5,
            warmup_lr_init=cfg.lr / 25,
            lr_min=cfg.lr / 500,
        )
        return [opt], [schd]
    else:
        raise NotImplementedError()


def resetmodel(model: nn.Module) -> None:
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(fn=weight_reset)
