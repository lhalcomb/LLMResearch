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



if __name__ == "__main__":

    #Applying a casual attention mask