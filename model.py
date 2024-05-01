import torch
from torch import nn
import torch.nn.functional as F


# collection of neurons that output only a single class value.
class subUnit(nn.Module):
    def __init__(self, inp_dim, depth=None, multi=1, unit=nn.Linear, bias=False):
        super().__init__()
        out_dim = inp_dim
        hid_dim = inp_dim * multi
        if not depth:
            depth = inp_dim

        self.unit = nn.ModuleList()
        self.unit.append(unit(inp_dim, hid_dim, bias=bias))
        self.unit.extend([unit(hid_dim, hid_dim, bias=bias) for i in range(depth - 2)])
        self.unit.append(unit(hid_dim, out_dim, bias=bias))

        self.unit.append(nn.Softmax(dim=-1))

    def forward(self, inp):
        out = self.unit(inp)
        out = F.one_hot(torch.argmax(out, dim=-1), num_classes=inp.shape[-1])
        return out


class LayerList(nn.Module):
    def __init__(self, dim, depth=None, sub_dim=2, sub_depth=2, bias=False):
        super().__init__()

        self.layer = nn.ModuleList(
            [subUnit(sub_dim, sub_depth, bias=bias) for i in range(dim)]
        )
        self.sub_dim = sub_dim

    def forward(self, inp):
        sdim = self.sub_dim
        B, TI = inp.shape
        oups = torch.zeros((B, TI))
        for uidx, unit in enumerate(self.layer):
            oups[:, sdim * uidx : sdim * (uidx + 1)] = unit(
                inp[:, sdim * uidx : sdim * (uidx + 1)]
            )
        return oups


class Unit(nn.Module):
    def __init__(self, dim, depth=None, sub_dim=2, sub_depth=2, bias=False):
        super().__init__()

        self.input = nn.Linear(dim * sub_dim, dim * sub_dim, bias=bias)
        self.layers = nn.ModuleList(
            [LayerList(dim, depth, sub_dim, sub_depth, bias) for d in range(depth)]
        )
        self.sub_dim = sub_dim

    def forward(self, inp):
        inp_inp = self.input(inp)  # B, inp_dim*sub_inp_dim
        B, TI = inp_inp.shape
        oups = torch.zeros((depth + 1, B, TI))
        oups[0] = inp_inp
        for depth, layer in enumerate(self.layers):
            oups[depth + 1] = layer(oups[depth])

        return oups[-1]
