
import torch 
from torch.nn import nn 

class LayerNormalization(): 
    def __init__(self, parameter_shape, eps=1e-5): 
        self.parameters_shape=parameter_shape
        self.eps=eps # to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(parameter_shape)) # gamma initiallized to 1 
        self.beta = nn.Parameter(torch.zeros(parameter_shape)) # beta initialized to zero 
        
    def forward(self, inputs): 
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs-mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std 
        out = self.gamma * y + self.beta 
        return out 