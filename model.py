import torch
from torch import nn
import torch.nn.functional as F


# collection of neurons that output only a single class value.
class subUnit(nn.Module):
    def __init__(self, inp_dim, hid_dim, depth, out_dim):
        super().__init__()

        self.unit = nn.ModuleList()
        self.unit.append(nn.Linear(inp_dim, hid_dim))
        self.unit.extend([nn.Linear(hid_dim, hid_dim) for i in range(depth)])
        self.unit.append(nn.Linear(hid_dim, out_dim))

        self.unit.append(nn.Softmax(dim=-1))

    def forward(self, inp):
        out = self.unit(inp)
        out = F.one_hot(torch.argmax(out, dim=-1), num_classes=inp.shape[-1])
        return out
