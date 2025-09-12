"""

A compact self attention class built for generalization of computing the keys, queries, 
values, attention scores, & attention weights. 


Description of Class: 

SelfAttention_v1 is a class derived from nn.Module, which is a
fundamental building block of PyTorch models that provides necessary functionalities
for model layer creation and management. --- you learn this in intro

__init__ method initializes trainable weight matrices (W_query, W_key, and
W_value) for queries, keys, and values, each transforming the input dimension d_in to
an output dimension d_out.

forward pass -- using the forward method, we compute the attention
scores (attn_scores) by multiplying queries and keys, normalizing these scores using
softmax.

Lastly -- we create a context vector by weighting the values with 
          these normalized attention scores. 
"""

import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key  = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T #omega wieghts
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim = -1       
        )

        context_vec = attn_weights @ values 

        return context_vec
    
        
