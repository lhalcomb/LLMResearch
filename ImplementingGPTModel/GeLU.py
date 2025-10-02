"""
This file implements the GELU, Gaussian error linear unit, activation function

"""

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
    



if __name__ == "__main__":
    gelu, relu, sigmoid = GELU(), nn.ReLU(), nn.Sigmoid()
    #This is for comparison between GeLU and ReLU and Sigmoid
    x = torch.linspace(-3, 3, 100) 
    y_gelu, y_relu, y_sigmoid = gelu(x), relu(x), sigmoid(x)
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu, y_sigmoid], ["GELU", "ReLU", "Sigmoid"]), 1):
        plt.subplot(1, 3, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()