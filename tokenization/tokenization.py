""" 
Credits to Sebastian Raschka's Build a Large Language Model From Scratch

Deep neural network models, including LLMs, cannot process raw text directly. Since
text is categorical, it isn't compatible with the mathematical operations used to implement
and train neural networks. Therefore, we need a way to represent words as
continuous-valued vectors.

Insert Embeddings -- 

At its core, an embedding is a mapping from discrete objects, such as words, images,
or even entire documents, to points in a continuous vector space—the primary purpose
of embeddings is to convert nonnumeric data into a format that neural networks
can process.

Different types of Embeddings:

Word, Sentence, Paragraph, etc.

"""
import re
from SimpleTokenizerV1 import SimpleTokenizerV1

if __name__ == "__main__":

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    

    """Practice with raw text tokenization"""
    # print(f"Total number of character: {len(raw_text)}" )
    # print(raw_text[:99])
    # text = "Hello, World! Is this -- a test?"
    # result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # result = [item.strip() for item in result if item.strip()]
    # print(result)

    """Fully preprocessed tokenization"""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(f"Length of preprocessed tokenized text: {len(preprocessed)} \n\n {preprocessed[:30]}")

    """Converting tokens into token IDs"""
    all_words = sorted(set(preprocessed)) #sort each word alphabetically in the text (no duplicates)
    vocab_size = len(all_words)
    print(vocab_size)

    vocab = {token:integer for integer, token in enumerate(all_words)}
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i >= 50:
    #         break

    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    convert_back = tokenizer.decode(ids)
    print(convert_back)

    """
    Code: 
    text = "Hello, do you like tea?"
    print(tokenizer.encode(text))

    This generates a KeyError because Hello does not show up in the vocabulary.
    Hence, highlighting the need to consider large and diverse training sets 
    to extend the vocabulary when working on LLMs.
    """
    
    """Adding special context tokens"""
    all_tokens = sorted(list(set(preprocessed))) 
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer, token in enumerate(all_tokens)}

    print(len(vocab.items()))
    #santiy check 
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    """Now let's try that again... """
    tokenizer = SimpleTokenizerV1(vocab)
    print(tokenizer.encode(text))

    #sanity check
    print(tokenizer.decode(tokenizer.encode(text)))

    """
    Depending on the LLM, some researchers also consider additional special tokens
    such as the following:
        -- [BOS] (beginning of sequence)—This token marks the start of a text. It signifies to
                the LLM where a piece of content begins.
        -- [EOS] (end of sequence)—This token is positioned at the end of a text and
                is especially useful when concatenating multiple unrelated texts, similar to
        <|endoftext|>. For instance, when combining two different Wikipedia articles
                or books, the [EOS] token indicates where one ends and the next begins.
        -- [PAD] (padding)—When training LLMs with batch sizes larger than one, the
                batch might contain texts of varying lengths. To ensure all texts have the same
                length, the shorter texts are extended or “padded” using the [PAD] token, up to
                the length of the longest text in the batch

    Note: When training on batch inputs, one typically uses a mask instead of padded tokens. 
    """