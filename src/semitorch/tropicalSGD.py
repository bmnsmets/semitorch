from torch import Tensor
from torch.linalg import vector_norm
from torch.optim import Optimizer


class TropicalSGD(Optimizer):
    """
    Corrects the learning rate of semiring related parameters
    by multiplying the learning rate in the update step
    with the norm of the input vector.

    If no input tensor is provided when the step() function is called,
    then this will default to standard SGD.

    Only recognizes the learning rate (lr; mandatory) parameter.
    """

    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(TropicalSGD, self).__init__(params, defaults)

    def step(self, input_tensor=Tensor | None, closure=None) -> None:
        if input_tensor is not None:
            norm_input = vector_norm(input_tensor)
        else:
            norm_input = 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.data, alpha=-group['lr'] * norm_input)
