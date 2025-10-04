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

import sys
import os

# Add the top-level project directory (/Users/laydenhalcomb/LLMResearch)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/laydenhalcomb/LLMResearch/LLMResearch")))

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
    #print(batch) #prints = (tensor([[6109, 3626, 6100,  345], [6109, 1110, 6622,  257]]))


    #next initialize the 124 million parameter DummyGPTModel
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)

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
    #print("Batch Example: ", batch_example)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    #print("Layers: ", layer)
    out = layer(batch_example)
    #print(out)

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
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    #now we calculate the standard deviation of the output layer
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    #print("Normalized layer outputs:\n", out_norm)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)


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
    #print("Mean: \n", mean)
    #print("Variance:\n", var)

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
    The FeedForward module plays a crucial role in enhancing the model's ability to learn
    from and generalize the data. Although the input and output dimensions of this
    module are the same, it internally expands the embedding dimension into a higherdimensional
    space through the first linear layer.
    
    This expansion is followed by a nonlinear GELU activation and then a contraction back to the original dimension with the second linear transformation. Such a design allows for the
    exploration of a richer representation space.


    See figure 4.10 on page 108 for better illustration of the linear layers
    """

    
    #4.3 Implementing a feed forward network with GELU activations



    from FeedForward import FeedForward
    ffn = FeedForward(GPT_CONFIG_124M)
    out = torch.rand(2, 3, 768)
    #print(out.shape) #prints 2, 3, 768


    #4.4 Adding Shortcut Connections 
    """
    Understanding the concept of Shortcut Connections: 

        Originally, shortcut connections were proposed for deep networks in
    computer vision (specifically, in residual networks) to mitigate the challenge of vanishing gradients. The vanishing gradient problem refers to the issue where gradients
    (which guide weight updates during training) become progressively smaller as they
    propagate backward through the layers, making it difficult to effectively train earlier
    layers.

    Shortcut conncetions in an LLM just creates an alternative, shorter path for the 
    griadent to flow thrugh the network by skipping one or more layers, which is achieved
    by adding the output of one layer to the output of a later layer. 

    This is why these connections are also known as skip connections. They play a crucial 
    role in preserving the flow of graidents during the backward pass in training. 


    Now we will implement shortcut connections in the feed forward method
    """

    from FeedForward import ExampleDeepNeuralNetwork

    layer_sizes = [3, 3, 3, 3, 3, 1] # layers are designed to 
    sample_input = torch.tensor([[1., 0., -1.]]) #accept an example with 3 input values
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork( 
    layer_sizes, use_shortcut=False
    )

    #here is the function that computes the gradients in the models backward pass

    def print_gradients(model, x):
        output = model(x) #forward pass
        target = torch.tensor([[0.]]) 
        loss = nn.MSELoss()
        loss = loss(output, target) #calculate loss on how close the target and output are
        loss.backward() #backward pass to calculate the gradients

        if model.use_shortcut == False: 
            print("Vanishing Gradients: \n")
        else: 
            print("Skip Connections Activated: \n")
        for name, param in model.named_parameters(): # we can loop through the weights via model.named_paramters()
            if 'weight' in name:
                print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

    #print_gradients(model_without_shortcut, sample_input)

    """
    Recall that the gradient vector  of a loss function indicates the direction of the steepest increase in error 
    for a given set of model parameters. By moving in the opposite (negative) direction 
    of this gradient, known as gradient descent, models can iteratively adjust their parameters to minimize the error, 
    leading to more accurate predictions and better performance. 

    Here that is exactly what we are trying to achieve but with a LLM. 

    Output: 
        layers.0.0.weight has gradient mean of 0.00020173587836325169
        layers.1.0.weight has gradient mean of 0.00012011159560643137
        layers.2.0.weight has gradient mean of 0.0007152039906941354
        layers.3.0.weight has gradient mean of 0.0013988736318424344
        layers.4.0.weight has gradient mean of 0.005049645435065031

        From the output above we see that the gradients become smaller
    as we progress from the last layer (layers.4) to the first layer (layers.0), which is
    a phenomenon called the vanishing gradient problem.

    """

    #we fix the vanishing gradient problem with the shortcut connection technique 
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
    )
    #print_gradients(model_with_shortcut, sample_input)


    """
        In conclusion, shortcut connections are important for overcoming the limitations
    posed by the vanishing gradient problem in deep neural networks. Shortcut connections are a core building block of very large models such as LLMs, and they will help
    facilitate more effective training by ensuring consistent gradient flow across layers
    when we train the GPT model in the next chapter.
    Next, we'll connect all of the previously covered concepts (layer normalization,
    GELU activations, feed forward module, and shortcut connections) in a transformer
    block, which is the final building block we need to code the GPT architecture.
    
    """

    #4.5 Connection attention and linear layers in a transformer block

    """
    This is where the fun begins. 

        We now are implementing the transformer block, a fundamental building block of GPT and
    other LLM architectures. This block, which is repeated a dozen times in the 124-millionparameter GPT-2 architecture, 
    combines several concepts we have previously covered:
        multi-head attention, layer normalization, dropout, feed forward layers, and GELU
        activations. 
    Later, we will connect this transformer block to the remaining parts of the
    GPT architecture
    
    """
    from TransformerBlock import TransformerBlock

    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    #print("Input shape:", x.shape)
    #print("Output shape:", output.shape)
    """
    Input shape: torch.Size([2, 4, 768])
    Output shape: torch.Size([2, 4, 768])
    """

    #4.6 Coding the GPT Model

    """
        We are now replacing the DummyTransformerBlock and DummyLayerNorm placeholders
    with the real TransformerBlock and LayerNorm classes we coded previously to assemble a fully working version of the original 124-million-parameter version of GPT-2. In
    chapter 5, we will pretrain a GPT-2 model, and in chapter 6, we will load in the pretrained weights from OpenAI
    """
    from GPTModel import GPTModel

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    """
    Output from above code: 

        tensor([[6109, 3626, 6100,  345],
            [6109, 1110, 6622,  257]])

    Output shape: torch.Size([2, 4, 50257])
    tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
            [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
            [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
            [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

            [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
            [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
            [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
            [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
        grad_fn=<UnsafeViewBackward0>)


        As we can see, the output tensor has the shape [2, 4, 50257], since we passed in two
    input texts with four tokens each. The last dimension, 50257, corresponds to the
    vocabulary size of the tokenizer. Later, we will see how to convert each of these 50,257-
    dimensional output vectors back into tokens.
    """

    #before we move forward, lets collect the total number of element params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


    total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
    )

    print(f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
    )

    """
        Weight tying reduces the overall memory footprint and computational complexity
    of the model. However, using separate token embedding and output layers results in better 
    training and model performance; so we use sepaarate layers in our GPTModel implementation
    
    Lastly, for 4.6, let's compute the memory requirements of the 163 million parameters in our
    GPTModel object
    """

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    #result: Total size of the model: 621.83 MB

    """
        In conclusion, by calculating the memory requirements for the 163 million parameters in our GPTModel object and assuming each parameter is a 32-bit float taking up 4
    bytes, we find that the total size of the model amounts to 621.83 MB, illustrating the
    relatively large storage capacity required to accommodate even relatively small LLMs. 
    
    """

    #4.7 Generating Text

