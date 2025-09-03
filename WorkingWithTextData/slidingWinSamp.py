"""
2.6 Data Sampling with a sliding window ----

Next step in creating embeddings for the LLM is to 
generate the input-target pairs required for training an LLM.
"""
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(f"Length of Encoded text: {len(enc_text)}")

    enc_sample = enc_text[50:] # cutoff for more interesting text passage... --No idea what this will do but i am intrigued (1)
    #print(enc_sample)

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1: context_size + 1]
    print(f"x: {x}")
    print(f"y:    {y}")

    """Next word prediction task"""

    """
        Everything left of the arrow (---->) refers to the input an LLM would receive, and
    the token ID on the right side of the arrow represents the target token ID that the
    LLM is supposed to predict.
    """
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    
    


