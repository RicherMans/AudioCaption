import numpy as np
import torch
import torch.nn as nn


class TimeMask(nn.Module):
    def __init__(self, n=1, p=50):
        super(TimeMask, self).__init__()
        self.p = p
        self.n = 1

    def forward(self, x):
        time, freq = x.shape
        if self.training:
            for i in range(self.n):
                t = torch.empty(1, dtype=int).random_(self.p).item()
                to_sample = max(time - t, 1)
                t0 = torch.empty(1, dtype=int).random_(to_sample).item()
                x[t0:t0+t, :] = 0
        return x

class FreqMask(nn.Module):
    def __init__(self, n=1, p=12):                                                                                                 
        super().__init__()
        self.p = p                                                                                                 
        self.n = 1                                                                                                 
                                                                                                    
    def forward(self, x):
        time, freq = x.shape
        if self.training: 
            for i in range(self.n):
                f = torch.empty(1, dtype=int).random_(self.p).item()
                f0 = torch.empty(1, dtype=int).random_(freq - f).item()                                                                                                 
                x[:, f0:f0 + f] = 0.                                                                                                 
        return x


if __name__ == "__main__":
    torch.manual_seed(1)
    timemask = FreqMask(1, 12)
    x = torch.randn(100, 64)
    timemask(x)
