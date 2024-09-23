import torch
from torch import nn

class ModelClass1(nn.Module):
    def __init__(self, 
                 input_shape: int, 
                 hidden_units1: int, 
                 hidden_units2: int, 
                 output_shape: int): 
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features= hidden_units1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=hidden_units1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=output_shape), 
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.layer_stack(x) 