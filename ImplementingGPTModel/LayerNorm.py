"""
In this file we have the normalized layer that we developed in the LLM architecture file


explanation of class: 

This specific implementation of layer normalization operates on the last dimension of
the input tensor x, which represents the embedding dimension (emb_dim). 
The variable eps is a small constant (epsilon) added to the variance to prevent division by zero
during normalization. The scale and shift are two trainable parameters (of the
same dimension as the input) that the LLM automatically adjusts during training if it
is determined that doing so would improve the model's performance on its training
task. This allows the model to learn appropriate scaling and shifting that best suit the
data it is processing
"""


import torch 
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #0.00001
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

"""
Comments on unbiased = False from author in textbook

Biased variance: 

In our variance calculation method, we use an implementation detail by setting
unbiased=False. For those curious about what this means, in the variance calculation, we divide by the number of inputs n in the variance formula. This approach does
not apply Bessel’s correction, which typically uses n – 1 instead of n in the denominator to adjust for bias in sample variance estimation. This decision results in a socalled biased estimate of the variance. For LLMs, where the embedding dimension n
is significantly large, the difference between using n and n – 1 is practically negligible.
I chose this approach to ensure compatibility with the GPT-2 model’s normalization
layers and because it reflects TensorFlow’s default behavior, which was used to
implement the original GPT-2 model. Using a similar setting ensures our method is
compatible with the pretrained weights we will load in chapter 6. 

"""