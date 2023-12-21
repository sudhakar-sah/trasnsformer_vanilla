
import torch 
import torch.nn as nn 

class LayerNormalization(nn.Module): 
    
    def __init__(self, parameters_shape, eps=1e-5): 
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps # to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # gamma initiallized to 1 
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # beta initialized to zero 
        
    def forward(self, inputs): 
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs-mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std 
        out = self.gamma * y + self.beta 
        return out 