"""
The beginnings of the LLM Architecture. 

This file will be the source file to the GPT placeholder architecture of Dummy-GPTModel,
to provide a big picture view of how everything will fit together so we can assemble 
the full GPT model architecture. 
"""
import torch
import torch.nn as nn
import tiktoken
from DummyGPTModel import DummyGPTModel
from LayerNorm import LayerNorm
tokenizer = tiktoken.get_encoding("gpt2")

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

"""
-- vocab_size refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
(see chapter 2).

-- context_length denotes the maximum number of input tokens the model can
handle via the positional embeddings (see chapter 2).

-- emb_dim represents the embedding size, transforming each token into a 768-
dimensional vector.

-- n_heads indicates the count of attention heads in the multi-head attention
mechanism (see chapter 3).

-- n_layers specifies the number of transformer blocks in the model, which we
will cover in the upcoming discussion.

-- drop_rate indicates the intensity of the dropout mechanism (0.1 implies a 10%
random drop out of hidden units) to prevent overfitting (see chapter 3).

-- qkv_bias determines whether to include a bias vector in the Linear layers of
the multi-head attention for query, key, and value computations. We will initially
disable this, following the norms of modern LLMs, but we will revisit it in chapter 6 when we load pretrained GPT-2 weights from OpenAI into our model (see
chapter 6)

"""

if __name__ == "__main__":

    #4.1 Coding an LLM architecture
    #Step1 - Implement dummy architecture



    #These are the batches of token IDs for each text string 
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch) #prints = (tensor([[6109, 3626, 6100,  345], [6109, 1110, 6622,  257]]))


    #next initialize the 124 million parameter DummyGPTModel
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

    #4.2 Normalizing activations with layer normalization
    """
    -- We will explore the layer normalization for normalizing activations. 
    This improves the stability and efficiency of neural network training. 

    -- We approach it with this idea:
            layer normalization is just the adjustment of the activations (outputs) 
            of a neural network layer to have a mean of 0 and a variance of 1, also
            known as unit variance.

    Note: 
    ReLU - Rectified Linear Unit: a standard activation function in neural networks that thresholds negative inputs to 0, 
    ensuring the layer only outputs positive values


    """

    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    print("Batch Example: ", batch_example)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    print("Layers: ", layer)
    out = layer(batch_example)
    print(out)

    """This above prints this below 

    Batch Example:  tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969],
        [ 0.2093, -0.9724, -0.7550,  0.3239, -0.1085]])

    Layers:  Sequential(
        (0): Linear(in_features=5, out_features=6, bias=True)
        (1): ReLU())

    tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)
    
    """
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    #now we calculate the standard deviation of the output layer
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    """
    output:

    Normalized layer outputs:
    tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
            [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
        grad_fn=<DivBackward0>)

    Mean:
    tensor([[-5.9605e-08],
            [ 1.9868e-08]], grad_fn=<MeanBackward1>)

    Variance:
    tensor([[1.0000],
            [1.0000]], grad_fn=<VarBackward0>)
    
    """



    #now lets apply our layer norm model in practice 

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim = -1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean: \n", mean)
    print("Variance:\n", var)

    """
    Result is below: 

    Mean: 
    tensor([[-2.9802e-08],
        [ 0.0000e+00]], grad_fn=<MeanBackward1>)
    Variance:
    tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
    
    """

    #next is looking at the GELU activation function 

    """
    
    
    """

    #4.3 Implementing a feed forward network with GELU activations