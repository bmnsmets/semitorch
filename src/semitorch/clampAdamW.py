from torch.optim import AdamW


class ClampAdamW(AdamW):
    """
    Clamps the updated parameters in a range [0, 1].
    """

    def step(self, closure=None) -> None:
        super().step(closure)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.clamp_(min=0, max=1)
