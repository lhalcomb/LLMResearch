import tiktoken 
import torch
from torch.utils.data import DataLoader
from GPTDatasetV1 import GPTDatasetV1


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


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # dataloader = create_dataloader_v1(
    #                 raw_text, 
    #                 batch_size=8, 
    #                 max_length=4, 
    #                 stride=4, 
    #                 shuffle=False
    #             )
    
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)

    # """The comments below for the respective lines apply when when parameters equal 1"""
    # """input tokens, target tokens"""
    # print(first_batch) #prints tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])] 
    # """printed the batch shifted right from above"""
    # second_batch = next(data_iter) #prints [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
    # print(second_batch)

    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print("Inputs:\n", inputs)
    # print("\nTargets:\n", targets)

    # """2.7 Creating token embeddings"""
    # vocab_size = 6
    # output_dim = 3
    # input_ids = torch.tensor([2, 3, 5, 1])
    # torch.manual_seed(123)
    # embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # print(embedding_layer.weight, "\n")
    # print(embedding_layer(input_ids))

    """2.8 Encoding word positions"""
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    """
    Input Processing Pipeline -----

        As part of the input processing pipeline, input text is first broken
    up into individual tokens. These tokens are then converted into token IDs using a
    vocabulary. The token IDs are converted into embedding vectors to which positional
    embeddings of a similar size are added, resulting in input embeddings that are used
    as input for the main LLM layers.
    """