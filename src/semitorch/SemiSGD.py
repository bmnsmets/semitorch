import torch


class SemiSGD(torch.optim.Optimizer):
    """
    Corrects the learning rate of semiring related parameters
    by multiplying the learning rate in the update step
    with the norm of the input vector.

    If no input tensor is provided when the step() function is called,
    then this will default to standard SGD.

    Only recognizes the learning rate (lr; mandatory) and
    weight decay (weight_decay; optional) parameters.
    """

    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SemiSGD, self).__init__(params, defaults)

    def step(self, input_tensor=torch.Tensor | None, closure=None) -> None:
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            if input_tensor is not None:
                norm_input = torch.norm(input_tensor)
            else:
                norm_input = 1

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-group['lr'] * norm_input, d_p)
