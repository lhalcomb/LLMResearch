"""
Here we will look at text generation strategies (also called decoding strategies) to generate
more original text. 

We will cover two techniques,
temperature scaling and top-k sampling, to improve this function.

"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ImplementingGPTModel"))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from GPTModel import GPTModel
from utility import generate_text_simple, text_to_token_ids, token_ids_to_text

tokenizer = tiktoken.get_encoding("gpt2")
model = GPT_CONFIG_124M = {
 "vocab_size": 50257,
 "context_length": 256,
 "emb_dim": 768,
 "n_heads": 12,
 "n_layers": 12,
 "drop_rate": 0.1,
 "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

model.to("cpu")
model.eval()


tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
 model=model,
 idx=text_to_token_ids("Every effort moves you", tokenizer),
 max_new_tokens=25,
 context_size=GPT_CONFIG_124M["context_length"]
)
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


#5.3.1 Temperature scaling

#previously we used greedy decoding in generate_text_simple using torch.argmax
#example for illustration

vocab = {
 "closer": 0,
 "every": 1,
 "effort": 2,
 "forward": 3,
 "inches": 4,
 "moves": 5,
 "pizza": 6,
 "toward": 7,
 "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
 [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])

torch.manual_seed(123)
#multinomial prob function
next_token_id = torch.multinomial(probas, num_samples=1).item()


def print_sampled_tokens(probas):
    torch.manual_seed(123)

    sample = [torch.multinomial(probas, num_samples=1).item()
            for i in range(1_000)]
    
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

    pizza_token = vocab["pizza"]
    print(f"\n'pizza' sampled {sampled_ids[pizza_token].item()} times.")

# print_sampled_tokens(probas)

#temp scaling - fancy description for dividing the logits by a number greater than 0

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# import matplotlib.pyplot as plt

# temperatures = [1, 0.1, 5]
# scaled_probas = [softmax_with_temperature(next_token_logits, T)
#                 for T in temperatures]
# x = torch.arange(len(vocab))
# bar_width = 0.15

# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(x + i * bar_width, scaled_probas[i],
#     bar_width, label=f'Temperature = {T}')
# ax.set_ylabel('Probability')
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
# plt.tight_layout()
# plt.show()

""" 
    Temperatures greater than 1 result in more uniformly distributed token probabilities,
    and temperatures smaller than 1 will result in more confident (sharper or more peaky)
    distributions.

    A temperature of 1 divides the logits by 1 before passing them to the softmax function to compute the probability scores.
    In other words, using a temperature of 1 is the same as not using any temperature scaling. In this case, the tokens are selected with a
    probability equal to the original softmax probability scores via the multinomial sampling function in  PyTorch. 
    For example, for the temperature setting 1, the token corresponding to “forward” would be selected about 60% of the time.  
    As shown in the matplot lib figure when you run this file. 

    Exercise 5.1
    Use the print_sampled_tokens function to print the sampling frequencies of the
    softmax probabilities scaled with the temperatures shown. How often
    is the word pizza sampled in each case? Can you think of a faster and more accurate
    way to determine how often the word pizza is sampled?
"""
# for T, probas in zip(temperatures, scaled_probas):
#     print(f"\n=== Temperature {T} ===")
#     print_sampled_tokens(probas)



#5.3.2 top-k sampling
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)
"""
Top logits: tensor([6.7500, 6.2800, 4.5100])
Top positions: tensor([3, 7, 0])
"""
print(top_logits[-1])
new_logits = torch.where(
 condition=next_token_logits < top_logits[-1],
 input=torch.tensor(float('-inf')),
 other=next_token_logits
)
print(new_logits)
#tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])

#turning these next tokens into probas
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas) #tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])




