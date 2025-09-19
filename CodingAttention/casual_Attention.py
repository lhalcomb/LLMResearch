"""
3.5 Hiding future words with casual attention 

For many LLM tasks, you will want the self-attention mechanism to consider only the tokens
that appear prior to the current position when predicting the next token in a sequence.
Casual attention, also known as masked attention, is a specialized form of self-attention.
It restricts a model to only consider previous and current inputs in asequence when processing 
any given token whenb computing attention scores. This is in contrast to the standard self-attention
mechanism, which allows access to the entire input sequence at once. 

Now we do a modification of the standard self-attention mechanism to create 
a casual attention mechanism, which is essential for developing an LLM. To achieve the 
GPT-like LLM architecure, for each token processed, we need to mask out the future tokens. These
future tokens come after the current token in the input text. 

We masl out the attention weights above the diagonal and we normalize the nonmasked
attention weights such that the attention weights can sum to 1 in each row. 
"""


import torch 
from SelfAttention_v2 import SelfAttention_v2


if __name__ == "__main__":

    #Applying a casual attention mask

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

    """
    We're reusing the query and key weight matrices
    of the SelfAttention_v2 object from the
    previous section for convenience
    """

    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)

    # print(attn_weights)
    # print(attn_scores)
    context_length = attn_scores.shape[0]
    # print(context_length)

    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)
    masked_simple = attn_weights * mask_simple
    print(masked_simple)

    row_sums = masked_simple.sum(dim = -1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums

    print(masked_simple_norm)

    """

    When applying a mask and then renormalizing the attention weights, at first it seems
    that the information from future tokens (which we tend to mask) could still influence
    the current token because their values are part of the softmax caclulation. 
    However, after masking and renormalization, the distribution of attention weights is
    as if it was calculated only among the unmasked positions to begin with. This ensures 
    that there isn't an information leak from future tokens as intended. 

    Why is this avoidance information leak important? 
    We want the model to predict the next word, not just generate the next token in a sequence. 
    """

    #A more efficient way to obtain the masked attention weight matrix in causal attention is 
    #to mask the attention scores with negative infinity values before
    #applying the softmax function.

    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked) #prints upper half of triangle with -inf intialized

    attn_weights = torch.softmax(masked / keys.shape[-1] ** 5, dim=1)
    print(attn_weights) #values in each row sum to 1, and no further normalization is necessary


    #3.5.2 Masking additional attention weights with dropout
    #another technique that is useful for reducing overfitting with the training of llms

    """
    Dropout - a deep learnig technique where randomly selected hidden layer units are ignored
    during training, effectively "dropping" them out. 
    This method helps prevent overfitting by ensuring that a model does not become 
    overly reliant on any specific set of hidden layer units.

    Dropout is only used during training, it is disabled afterward. 
    """

    #example of drop out
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))

    #applying dropout to the attention matrix itself
    torch.manual_seed(123)
    print(dropout(attn_weights))

    """
    Note that the resulting dropout outputs may look different depending on your operating
    system; you can read more about this inconsistency here on the PyTorch issue
    tracker at https://github.com/pytorch/pytorch/issues/121595.

    Having gained an understanding of causal attention and dropout masking, we can
    now develop a concise Python class. This class is designed to facilitate the efficient
    application of these two techniques.
    """

    #first lets see if the code can handle batches of more than one input for the sake of support of our new class

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) #torch.Size([2, 6, 3]) 3d tensor consisting of two input texts with 6 tokens each s.t. each is a 3d embedding vector
    
    