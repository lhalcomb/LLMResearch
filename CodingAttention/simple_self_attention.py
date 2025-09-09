""" Heavily documented because of complexity.
Here we will introduce a simplified self-attention model to understand the 
idea of self-attention and what that means. 

Note: 
    Before transformers and LLMs, researchers used Recurrent Neural Networks (RNN) 
to encode and decode text data from their embeddings or tokens. This idea is great, for 
very small amounts of texts. However, when feeding a RNN a large amount of data, the model 
can begin to lose a lot of context. So attention mechanisms were built into this RNN. 

Some time later (Circa 2014) these researchers used a technique called "Bahdanau attention" mechanism.
This technique would modify the encoder-decoder part of the RNN such that the decoder
could selectively access different parts of the input sentence of each decoding step. 
Essentially giving the RNN a context-senstive understanding of text based inputs. 

A paper published three years later found that the RNN was complete boloney and all 
you needed was this attention mechanism. Thus, transformers were born and the world of 
natural language processing has forever changed. 

Which had led me to the interest of LLMs... thanks researchers at Google. 

Probably the most important paper of the 21st century. 
See paper at this link: https://arxiv.org/abs/1706.03762

"""


import torch

"""
What the hell is self-attention? 

In self-attention, the “self” refers to the mechanism's ability to compute attention
weights by relating different positions within a single input sequence. It assesses and
learns the relationships and dependencies between various parts of the input itself,
such as words in a sentence or pixels in an image.

This is in contrast to traditional attention mechanisms, where the focus is on the relationships
between elements of two different sequences, such as in sequence-to-sequence
models where the attention might be between an input sequence and an
output sequence

"""

#Pay close attention... Pun the author made

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

if __name__ == "__main__":

    """
    Below we have the following input sentence which has already been broken down
    into 3-dimensional vectors. 
    (See Build a Large Language Model from Scratch pg. 57-60 for corresponding diagram)
    
    """

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )

    # Step 1: compute the intermediate values, omega, referred to as attention scores.
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])

    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    
     # very simple linear algebra. Compute dot product of "query" input with every input token
    print(attn_scores_2) #result tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

    """ Last big note: 
            Beyond viewing the dot product operation as a mathematical tool that combines
        two vectors to yield a scalar value, the dot product is a measure of similarity
        because it quantifies how closely two vectors are aligned: a higher dot product indicates
        a greater degree of alignment or similarity between the vectors. In the context
        of self-attention mechanisms, the dot product determines the extent to which
        each element in a sequence focuses on, or “attends to,” any other element: the
        higher the dot product, the higher the similarity and attention score between two
        elements.
    """

    #Step 2: normalization 
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    #typically the use of softmax is better.
    
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())

    #final step: calculate the context vector (z^2). See pg. 61
    attn_weights_2 = attn_weights_2_naive
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)


    #now we are ready to generalize this procedure for computing context vectors to calculate all context vectors simultaneously - how exciting. 

    #Computing attention weights for all input tokens

    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    # print(attn_scores)

    #results are as follows
    """
    tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
    """

    #you see the for loops above? Forget them. We're gonna use matrix multiplication between pytorch tensors
    #step 1 done better
    attn_scores = inputs @ inputs.T  #what in the high level malarkey
    #produces previous results with less clutter. Does same thing too 
    print(attn_scores) 


    #step 2: normalize this above. Guass would be proud
    attn_weights = torch.softmax(attn_scores, dim=1)
    print(attn_weights)
    # normalized results are as follows
    """
    tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
    short documentation for softmax(inputs, dim=blah) here: 
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    """

    #verification of rows summing to 1
    print("All row sums: ", attn_weights.sum(dim=1))

    #3rd and final step: matrix mul on the weights to compute ALL context vectors
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)

    """
    result: 
        tensor([[0.4421, 0.5931, 0.5790],
                [0.4419, 0.6515, 0.5683],
                [0.4431, 0.6496, 0.5671],
                [0.4304, 0.6298, 0.5510],
                [0.4671, 0.5910, 0.5266],
                [0.4177, 0.6503, 0.5645]])
    """

    print("Previous 2nd context vector:", context_vec_2) # sanity check

    """
        This concludes the code walkthrough of a simple self-attention mechanism. Next, we
    will add trainable weights, enabling the LLM to learn from data and improve its performance
    on specific tasks.
    """

    