"""
Byte Pair Encoding (BPE), the tokenizer that was originally 
used to train LLMs such as GPT-2, GPT-3, and the original model used in ChatGPT.

tiktoken - open source Python library that implements the BPE algorithm in Rust. 
Repository Here: https://github.com/openai/tiktoken

"""
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    """
    Just like our SimpleTokenizerV1 class, this BPE implementation is an efficient way
    to tokenize text. 
    """
    text = (
            "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
            "of someunknownPlace."
        )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)

    text2 = "Akwirw ier"
    integers2 = tokenizer.encode(text2)
    print(integers2)

    strings2 = tokenizer.decode(integers2)
    print(strings2)