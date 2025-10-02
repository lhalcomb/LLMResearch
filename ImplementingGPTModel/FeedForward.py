"""
this file uses the GeLU function previously created to develop a 
small neural network module, FeedForward, for training in the LLM transformer block later



"""
import torch
import torch.nn as nn
from GeLU import GELU

class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    )
        
    def forward(self, x):
        return self.layers(x)

"""

The code implements a deep neural network with five layers, each consisting of a
Linear layer and a GELU activation function. In the forward pass, we iteratively pass the
input through the layers and optionally add the shortcut connections if the self.use_
shortcut attribute is set to True. 

"""
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut # below we have five sequntial linear layers
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                                    GELU()),
                                    nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                                    GELU()),
                                    nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                                    GELU()),
                                    nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                                    GELU()),
                                    nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                                    GELU())
                                    ])
        
    def forward(self, x):
        for layer in self.layers: #here we compute the output of the current layer 
            layer_output = layer(x) #Compute here
            if self.use_shortcut and x.shape == layer_output.shape: #here we check if the shortcut can be applied
                x = x + layer_output 
            else:
                x = layer_output
        return x