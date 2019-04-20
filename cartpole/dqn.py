import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out)
        )
        self.init_weights()

    def forward(self, x):
        return self.net.forward(x)

    def np_forward(self, x):
        return self.forward(torch.FloatTensor(x).unsqueeze(0))

    def init_weights(self):
        for p in self.parameters():
            nn.init.normal_(p)
