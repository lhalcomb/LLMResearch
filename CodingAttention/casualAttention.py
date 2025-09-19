"""
The following CausalAttention class is similar to the SelfAttention class we implemented
earlier, except that we added the dropout and causal mask components



register_buffer() -- this in PyTorch is not strictly necessary for all use cases but offers several advantages here. 
                For instance, when we use the CausalAttention class in our LLM, buffers are automatically
                moved to the appropriate device (CPU or GPU) along with our model, which will
                be relevant when training our LLM. This means we don't need to manually ensure
                these tensors are on the same device as your model parameters, avoiding device mismatch
                errors.
"""

import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        #Compared to the previous SelfAttention_v1 class, we added a dropout layer.

        self.register_buffer( #The register_buffer call is also a new addition (more information is provided in the following text).
        'mask',
        torch.triu(torch.ones(context_length, context_length),
        diagonal=1)
        )



    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        """
        We transpose
        dimensions 1 and 2,
        keeping the batch
        dimension at the first
        position (0).
        """

        attn_scores.masked_fill_( 
        #In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies.

        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1

        )
        
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec
    





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
    
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape) #boom! context_vecs.shape: torch.Size([2, 6, 2])


    #next we will expand on this concept into a 
    #multi-head attention module that implements several 
    #casual attentions in parallel