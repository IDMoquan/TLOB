import torch
import torch.nn as nn

class GatingLayer(nn.Module):
    def __init__(self, input_dim: int):
        super(GatingLayer, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        gate = self.sigmoid(self.linear(x))
        return x * gate