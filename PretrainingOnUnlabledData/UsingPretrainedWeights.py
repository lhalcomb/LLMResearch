"""
Thus far, we have discussed how to numerically evaluate the training progress and pretrain an LLM from scratch. Even though both the LLM and dataset were relatively
small, this exercise showed that pretraining LLMs is computationally expensive. Thus,
it is important to be able to save the LLM so that we don't have to rerun the training
every time we want to use it in a new session.
So, let's discuss how to save and load a pretrained model. 
Later, we will load a more capable pretrained GPT model from OpenAI into our GPTModel instance. 

"""

"""
Saving training from LLM using pytorch


#torch.save(model.state_dict(), "model.pth") #generic use

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

torch.save({
 "model_state_dict": model.state_dict(),
 "optimizer_state_dict": optimizer.state_dict(),
 },
 "model_and_optimizer.pth"
)

#restore weights
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();

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
from utility import generate, text_to_token_ids, token_ids_to_text, train_model_simple
tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_CONFIG_124M = {
 "vocab_size": 50257,
 "context_length": 256, #1024
 "emb_dim": 768,
 "n_heads": 12,
 "n_layers": 12,
 "drop_rate": 0.1,
 "qkv_bias": False #True
}


#5.5 Loading Pretrained Weights from OpenAI 

from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
 model_size="124M", models_dir="gpt2"
)

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)


# Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
# [[-0.11010301 -0.03926672  0.03310751 ... -0.1363697   0.01506208
#    0.04531523]
#  [ 0.04034033 -0.04861503  0.04624869 ...  0.08605453  0.00253983
#    0.04318958]
#  [-0.12746179  0.04793796  0.18410145 ...  0.08991534 -0.12972379
#   -0.08785918]
#  ...
#  [-0.04453601 -0.05483596  0.01225674 ...  0.10435229  0.09783269
#   -0.06952604]
#  [ 0.1860082   0.01665728  0.04611587 ... -0.09625227  0.07847701
#   -0.02245961]
#  [ 0.05135201 -0.02768905  0.0499369  ...  0.00704835  0.15519823
#    0.12067825]]
# Token embedding weight tensor dimensions: (50257, 768)

"""
We downloaded and loaded the weights of the smallest GPT-2 model via the download_
and_load_gpt2(model_size="124M", ...) setting. OpenAI also shares the weights of
larger models: 355M, 774M, and 1558M. 
"""

model_configs = {
 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
NEW_CONFIG.update(model_configs[model_name])

gpt = GPTModel(NEW_CONFIG) 
gpt.eval()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


import numpy as np
def load_weights_into_gpt(gpt, params): #Sets the model’s positional and token embedding weights to those specified in params.
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        """
            The np.split function is used to divide the attention and bias weights
    into three equal parts for the query, key, and value components.
        """
        q_w, k_w, v_w = np.split( #Iterates over each transformer block in the model
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
        gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
        gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
        gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
        gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
        gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
        gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
        gpt.trf_blocks[b].att.out_proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
        gpt.trf_blocks[b].att.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
        gpt.trf_blocks[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
        gpt.trf_blocks[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
        gpt.trf_blocks[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
        gpt.trf_blocks[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
        gpt.trf_blocks[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
        gpt.trf_blocks[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
        gpt.trf_blocks[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
        gpt.trf_blocks[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])

    """
        The original GPT-2 model
        by OpenAI reused the token
        embedding weights in the
        output layer to reduce the
        total number of parameters,
        which is a concept known as
        weight tying.
    """
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) 


#The GPTModel instance is now initialized with pretrained weights from OpenAI
load_weights_into_gpt(gpt, params)
gpt.to(device)


file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters) #20,479
print("Tokens:", total_tokens) #5,145

dataloaderPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ch2_WorkingWithTextData"))
sys.path.append(dataloaderPath)

from GPTDatasetV1 import GPTDatasetV1
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=num_workers
                )
    return dataloader

torch.manual_seed(123)
train_loader = create_dataloader_v1(
 train_data,
 batch_size=2,
 max_length=GPT_CONFIG_124M["context_length"],
 stride=GPT_CONFIG_124M["context_length"],
 drop_last=True,
 shuffle=True,
 num_workers=0
)
val_loader = create_dataloader_v1(
 val_data,
 batch_size=2,
 max_length=GPT_CONFIG_124M["context_length"],
 stride=GPT_CONFIG_124M["context_length"],
 drop_last=False,
 shuffle=False,
 num_workers=0
)
optimizer = torch.optim.AdamW(
 gpt.parameters(),
 lr=0.0004, weight_decay=0.1
)


num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
 gpt, train_loader, val_loader, optimizer, device,
 num_epochs=num_epochs, eval_freq=5, eval_iter=5,
 start_context="Every effort moves you", tokenizer=tokenizer
)

old_prompt = "Every effort moves you"
new_prompt = "Hi, My name is"

torch.manual_seed(123)
token_ids = generate(
 model=gpt,
 idx=text_to_token_ids(new_prompt, tokenizer).to(device),
 max_new_tokens=25,
 context_size=NEW_CONFIG["context_length"],
 top_k=50,
 temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))