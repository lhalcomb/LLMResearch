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
    print(len(enc_text))

    enc_sample = enc_text[50:] # cutoff for more interesting text passage... --No idea what this will do but i am intrigued (1)
    #print(enc_sample)

    
