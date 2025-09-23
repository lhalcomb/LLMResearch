"""
3.6 Extending single-head attention to multi-head attention

Let's extend the previously implemented casual attention class over multiple heads. 
Hence the name -- multihead attention

"""

#3.6.1 stacking multiple single-head attention layers 

"""
The multi-head attention module includes two single-head attention modules stacked on top of
each other. So, instead of using a single matrix Wv for computing the value matrices, in a multi-head attention
module with two heads, we now have two value weight matrices: Wv1 and Wv2. The same applies to the other
weight matrices, WQ and Wk. We obtain two sets of context vectors Z1 and Z2 that we can combine into a single
context vector matrix Z.

"""

#Lets create a wrapper class to implement multi-head attention
from CodingAttention.CausalAttention import CausalAttention
import torch
import torch.nn as nn

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length,
        dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
                )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    


if __name__ == "__main__":

    torch.manual_seed(123)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )

    d_in = inputs.shape[1] # the embedding size. d_in = 3 bc [0.55, 0.87, 0.66], # journey (x^2)
    d_out = 2 # hardcoded output embedding size. typically equal to input but for illustration purposes we will go with 2

    batch = torch.stack((inputs, inputs), dim=0)
    
    context_length = batch.shape[1] # This is the number of tokens
    d_in, d_out = 3, 2 #1

    mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
    )

    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    #quick exercise 
    """
        Change the input arguments for the MultiHeadAttentionWrapper(..., num_
    heads=2) call such that the output context vectors are two-dimensional instead of
    four dimensional while keeping the setting num_heads=2. Hint: You don't have to
    modify the class implementation; you just have to change one of the other input
    arguments.
        
    we change d_out = 1
    """

    

