import re

"""
tokenizer class --- SimpleTokenizerV1
 encode --- splits text into tokens and carries out the 
            string-to-integer mapping to produce token
            IDs via the vocabulary. 

 decode --- carries out the reverse integer-to-string mapping 
            to convert the token IDs back into text.

"""


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        """Processes input text into token IDs"""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids
    
    def decode(self, ids):
        """Converts token IDs back into text"""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #Removes spaces before the specified puncuation
        return text 