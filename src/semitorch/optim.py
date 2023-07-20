class MultiOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

class MultiLRScheduler:
    def __init__(self, *sch):
        self.schedulers = sch

    def step(self, *args, **kwargs):
        for s in self.schedulers:
            s.step(*args, **kwargs)
