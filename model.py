import torch
import torch.nn as nn
import torch.nn.functional as F

# feed forward neural network
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.l1 = nn.Linear(194, 1024)
        self.r1 = nn.ReLU()
        self.d1 =nn.Dropout(0.2)

        self.l2 = nn.Linear(1024, 512)
        self.r2 = nn.ReLU()
        self.d2 =nn.Dropout(0.2)

        self.l3 = nn.Linear(512, 128)
        self.r3 = nn.ReLU()
        self.d2 =nn.Dropout(0.2)

        self.l4 = nn.Linear(128, 2)
        self.out = nn.Sigmoid()


    def forward(self, x):
        l1 = self.l1(x)
        r1 = self.r1(l1)
        d1 = self.d1(r1)

        l2 = self.l2(d1)
        r2 = self.r2(l2)
        d2 = self.d1(r2)

        l3 = self.l3(d2)
        r3 = self.r3(l3)
        d3 = self.d1(r3)

        l4 = self.l4(d3)
        y = self.out(l4)
        return y
