#3.6.2 Implementing multi-head attention with weight splits 
"""
    In addition to merging the MultiHeadAttentionWrapper with the Causal-
    Attention code, we will make some other modifications to implement multi-head
    attention more efficiently by developing a multiheadattention class
    
"""

import torch 
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
        context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #Reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # uses a linear layer to combine head inputs
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_length, context_length),
        diagonal=1)
        )

    def forward(self, x):
        #Tensor Shape: b, num_tokens, d_out
        b, num_tokens, d_in = x.shape

        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        """
        We implicitly split the matrix by adding a num_heads dimension. 
        Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        """
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
        b, num_tokens, self.num_heads, self.head_dim
        )

        """
        Transposes from shape (b, num_tokens,
        num_heads, head_dim) to (b, num_heads,
        num_tokens, head_dim)
        """
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) #Computes the dot product for each head

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # masks truncated the number of tokens

        attn_scores.masked_fill_(mask_bool, -torch.inf) #uses the mask to fill attention scores

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        attn_weights = self.dropout(attn_weights)

        #Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        #Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec) #Adds an optional linear projection

        return context_vec
    

"""
    In the MultiHeadAttentionWrapper class with two attention heads,
we initialized two weight matrices, Wq1 and Wq2, and computed two query matrices, Q1
and Q2 (top). In the MultiheadAttention class, we initialize one larger weight matrix
Wq, only perform one matrix multiplication with the inputs to obtain a query matrix Q, and
then split the query matrix into Q1 and Q2 (bottom). We do the same for the keys and
values, which are not shown to reduce visual clutter.
"""


if __name__ == "__main__":

    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],# the shape of this tensor
                        [0.8993, 0.0390, 0.9268, 0.7388],# is (b, num_heads, num_tokens, head_dim)
                        [0.7179, 0.7058, 0.9156, 0.4340]], # = (1, 2, 3, 4)
                        [[0.0772, 0.3565, 0.1479, 0.5331],
                        [0.4066, 0.2318, 0.4545, 0.9737],
                        [0.4606, 0.5159, 0.4220, 0.5786]]]]) 
    
    print(a @ a.transpose(2, 3)) # batch matrix multiplication

    # these results will be the exact same as those obtained when using the batched matrix mult above
    first_head = a[0, 0, :, :] # compute 
    first_res = first_head @ first_head.T # for multiple
    print("First head:\n", first_res) # heads separately

    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("\nSecond head:\n", second_res)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )

    batch = torch.stack((inputs, inputs), dim=0)

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    print(batch.shape)
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


    """
    And there you have it. We have implemented the attention mechanism used in LLMs

        With the  MultiHeadAttention class, we will use when we completely
    implement and train the LLM. 
    Note that while the code is fully functional, I used relatively small embedding sizes and numbers of attention heads to keep the outputs
    readable.

        For comparison, the smallest GPT-2 model (117 million parameters) has 12 attention
    heads and a context vector embedding size of 768. The largest GPT-2 model (1.5
    billion parameters) has 25 attention heads and a context vector embedding size of
    1,600. The embedding sizes of the token inputs and context embeddings are the same
    in GPT models (d_in = d_out).
        
    
    """


    """
    Exercise 3.3 Initializing GPT-2 size attention modules

    Using the MultiHeadAttention class, initialize a multi-head attention module that
    has the same number of attention heads as the smallest GPT-2 model (12 attention
    heads). Also ensure that you use the respective input and output embedding sizes
    similar to GPT-2 (768 dimensions). Note that the smallest GPT-2 model supports a
    context length of 1,024 tokens.
    
    """

    #initialize 12 attention heads 
    batch_size = 2
    context_length = 6  # or 1024 for full GPT-2 context
    d_in = 768

    batch = torch.randn(batch_size, context_length, d_in)
    d_out = 768
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    """
    Output to exercise: 
    
    tensor([[[ 0.1938, -0.4028,  0.2535,  ..., -0.4844, -0.0307,  0.3891],
         [-0.3292, -0.2296, -0.3402,  ..., -0.0953, -0.0705,  0.2572],
         [-0.1594, -0.2383, -0.2637,  ..., -0.3061, -0.2265,  0.2588],
         [-0.1648, -0.2645, -0.2489,  ..., -0.1302, -0.1685, -0.0502],
         [-0.1061, -0.1696, -0.3071,  ..., -0.1440, -0.1882,  0.0031],
         [-0.0540, -0.0980, -0.2584,  ..., -0.1585, -0.1064,  0.0659]],

        [[-0.2198,  0.1891, -0.3497,  ...,  0.6095,  0.2681, -0.6041],
         [-0.0686,  0.0326, -0.3965,  ...,  0.5302,  0.3781, -0.2824],
         [-0.0534,  0.1904, -0.2751,  ...,  0.1723,  0.3515, -0.2601],
         [-0.0588,  0.0993, -0.2948,  ...,  0.2962,  0.3626, -0.2333],
         [ 0.1758, -0.0054, -0.3116,  ...,  0.1285,  0.2854, -0.2940],
         [ 0.1984, -0.0383, -0.1473,  ...,  0.0538,  0.3011, -0.3956]]],
       grad_fn=<ViewBackward0>)
context_vecs.shape: torch.Size([2, 6, 768])

    """