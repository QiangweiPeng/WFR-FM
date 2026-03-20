import torch


class EMA:
    def __init__(self, model, decay):
        if not (0.0 < decay < 1.0):
            raise ValueError("Decay must be between 0 and 1.")

        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name not in self.shadow:
                        raise KeyError(f"Parameter {name} was not registered for EMA")
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
