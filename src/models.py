import torch
import torch.nn as nn


__all__ = [
    "velocityNet",
    "growthNet",
    "scoreNet",
    "dediffusionNet",
    "indediffusionNet",
    "FNet",
    "ODEFunc2",
]


class velocityNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation="Tanh"):
        super().__init__()
        layers = [in_out_dim + 1]
        for _ in range(n_hiddens):
            layers.append(hidden_dim)
        layers.append(in_out_dim)

        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1]),
                    self.activation,
                )
                for i in range(len(layers) - 2)
            ]
        )
        self.out = nn.Linear(layers[-2], layers[-1])

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)

        for idx, layer in enumerate(self.net):
            x = layer(state if idx == 0 else x)
        return self.out(x)


class growthNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)
        return self.net(state)


class scoreNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)
        return self.net(state)


class dediffusionNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state = torch.cat((t, x), dim=1)
        return self.net(state)


class indediffusionNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        return self.net(t)


class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()

        return v, g, s, d


class ODEFunc2(nn.Module):
    def __init__(self, f_net):
        super().__init__()
        self.f_net = f_net

    def forward(self, t, state):
        z, _ = state
        outputs = self.f_net(t, z)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            v, g = outputs[:2]
        else:
            raise ValueError("f_net must return at least velocity and growth outputs")
        return v.float(), g.float()
