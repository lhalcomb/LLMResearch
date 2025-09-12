"""
3.4 Imlementing self-attention with trainable weights

Same logic as simplified_self_attention.py, but this time we are extending the implementation 
of self attention to trainable weights. 

scaled dot-product attention --- the self-attention mechanism used in the orignal transformer architecture. 

Tackling this self-attention mechanism in the two subsections: 
    1. we will code the self-attention mechanism step by step as before. 
    2. we will organize the code into a compact Python class
       that can be imported into the LLM architecture.

Note: To better understand why the dot product is so important for machine learning, take a Linear Regression course.
      Or buy a textbook on the subject.

next --> casual attention

"""
import torch
from SelfAttention_v1 import SelfAttention_v1
from SelfAttention_v2 import SelfAttention_v2
torch.manual_seed(123)



if __name__ == "__main__": 
    """

    The code here only computes one context vector z^(2) for understanding the computation of attention weights.
    Then we will make a class to generalize this for each context vector
    
    """

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )

    x_2 = inputs[1] #the second input element "journey"
    d_in = inputs.shape[1] # the embedding size. d_in = 3 bc [0.55, 0.87, 0.66], # journey (x^2)
    d_out = 2 # hardcoded output embedding size. typically equal to input but for illustration purposes we will go with 2

    #Initialize the three weight matrices for query, key, & value
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

    """
        We set requires_grad=False to reduce clutter in the outputs, but if we were to use
    the weight matrices for model training, we would set requires_grad=True to update
    these matrices during model training.
    """

    print(W_query)

    #Next, we compute the query, key, and value vectors

    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)

    """
        Weight parameters vs. attention weights
    In the weight matrices W, the term “weight” is short for “weight parameters,” the values
    of a neural network that are optimized during training. This is not to be confused
    with the attention weights. As we already saw, attention weights determine the extent
    to which a context vector depends on the different parts of the input (i.e., to what
    extent the network focuses on different parts of the input).
    In summary, weight parameters are the fundamental, learned coefficients that define
    the network's connections, while attention weights are dynamic, context-specific values
    """

    """
        Note: Even though our temporary goal is only to compute the one context vector, z(2), we still
    require the key and value vectors for all input elements as they are involved in computing
    the attention weights with respect to the query q(2)
    """
    
    #we can obtain all the keys via matrix multiplication 
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)


    #now we compute the attention scores

    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22) # the result here for this attentionscore unnormalized is tensor(1.8524)

    #we can generalize this computation to all attention scores via matrix multiplication just like before

    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)

    #tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

    #now we go from attenion scores to attention weights
    """
            We compute the attention weights by scaling the attention scores and
    using the softmax function. However, now we scale the attention scores by dividing
    them by the square root of the embedding dimension of the keys (taking the square
    root is mathematically the same as exponentiating by 0.5):
    """

    d_k = keys.shape[-1]
    print(keys.shape[-1])

    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim = -1) # why does the author use -1, its the same as 0? 
    print(attn_weights_2)

    """
    
    The rationale behind scaled-dot product attention:

    The reason for the normalization by the embedding dimension size is to improve the
    training performance by avoiding small gradients. For instance, when scaling up the
    embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large
    dot products can result in very small gradients during backpropagation due to the
    softmax function applied to them. As dot products increase, the softmax function
    behaves more like a step function, resulting in gradients nearing zero. These small
    gradients can drastically slow down learning or cause training to stagnate.
    The scaling by the square root of the embedding dimension is the reason why this
    self-attention mechanism is also called scaled-dot product attention.
    """

    #final step is compute all of the context vectors

    #similar to when we computed the context vector as a weighted sum over the input vectors
    #we now compiute teh context vector as a weighted sum over the value vectors
    #here, the attention weights act like a weighting factor that weighs the respective importance 
    #of each value vector. Here, we can use matrix multiplication just like before: 

    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)

    # and this computes the general context vector z^2


    """
    
    Aside:  Why query, key, and value?

    The terms “key,” “query,” and “value” in the context of attention mechanisms are
    borrowed from the domain of information retrieval and databases, where similar concepts
    are used to store, search, and retrieve information.
    A query is analogous to a search query in a database. It represents the current item
    (e.g., a word or token in a sentence) the model focuses on or tries to understand.
    The query is used to probe the other parts of the input sequence to determine how
    much attention to pay to them.
    The key is like a database key used for indexing and searching. In the attention mechanism,
    each item in the input sequence (e.g., each word in a sentence) has an associated
    key. These keys are used to match the query.
    The value in this context is similar to the value in a key-value pair in a database. It
    represents the actual content or representation of the input items. Once the model
    determines which keys (and thus which parts of the input) are most relevant to the
    query (the current focus item), it retrieves the corresponding values.
    """


    #---> Implementing a compact self-attention Python class: class is in own file but usage will be shown below


    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    #output of sa_v1(inputs)
    """
    tensor([[1.4035, 1.0391],
        [1.4410, 1.0669],
        [1.4391, 1.0655],
        [1.3786, 1.0178],
        [1.3653, 1.0086],
        [1.4025, 1.0361]], grad_fn=<MmBackward0>)
    """

    """To summarize what we have done 
        Self-attention involves the trainable weight matrices Wq, Wk, and Wv. These matrices
    transform input data into queries, keys, and values, respectively, which are crucial components
    of the attention mechanism.
    
    """

    #Now we will fix the class to get rid of the nn.Parameter(torch.rand(...) approach and replace it with linear layers 

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    # refer to pg.72 figure 3.18 for model of what we have done thus far

    """
    Note: 
        SelfAttention_v1 and SelfAttention_v2 give different outputs because
    they use different initial weights for the weight matrices since nn.Linear uses a more
    sophisticated weight initialization scheme.
    """

    #quick exercise 3.1 Comparing SelfAttention_v1 and SelfAttention_v2

    """
        Note that nn.Linear in SelfAttention_v2 uses a different weight initialization
    scheme as nn.Parameter(torch.rand(d_in, d_out)) used in SelfAttention_v1,
    which causes both mechanisms to produce different results. To check that both
    implementations, SelfAttention_v1 and SelfAttention_v2, are otherwise similar,
    we can transfer the weight matrices from a SelfAttention_v2 object to a Self-
    Attention_v1, such that both objects then produce the same results.

    Your task is to correctly assign the weights from an instance of SelfAttention_v2
    to an instance of SelfAttention_v1. To do this, you need to understand the relationship
    between the weights in both versions. (Hint: nn.Linear stores the weight
    matrix in a transposed form.) After the assignment, you should observe that both
    instances produce the same outputs.
    
    """

    

    """
    Next ---> 
        We will make enhancements to the self-attention mechanism, focusing specifically
    on incorporating causal and multi-head elements. The causal aspect involves modifying
    the attention mechanism to prevent the model from accessing future information
    in the sequence, which is crucial for tasks like language modeling, where each word
    prediction should only depend on previous words.

    The multi-head component involves splitting the attention mechanism into multiple
    “heads.” Each head learns different aspects of the data, allowing the model to
    simultaneously attend to information from different representation subspaces at different
    positions. This improves the model's performance in complex tasks
    
    """