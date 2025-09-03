# Small Language Model (SLM)

This project is an experimental implementation of a **Small Language Model (SLM)** built from scratch. The goal is to learn and demonstrate the core concepts behind training and running large language models—on a smaller scale that is easier to understand, train, and run on consumer hardware.  

## Features
- Tokenization and vocabulary building  
- Embedding layers for token representation  
- Transformer-style architecture (multi-head attention + feedforward)  
- Training loop with loss tracking  
- Sampling and text generation  
- Configurable hyperparameters (layers, hidden size, vocab size, etc.)  

## Learning Goals
- Understand how tokenization and embeddings work  
- Implement attention and transformer blocks from scratch  
- Train a small but functional autoregressive model  
- Explore scaling laws and limitations of small models  

## Project Goals 
- Potential About me chat feature on Portfolio Website
  



## Byte Pair Encoding 

The algorithm underlying BPE breaks down words that aren’t in its predefined
vocabulary into smaller subword units or even individual characters, enabling it to
handle out-of-vocabulary words.
In short, it builds its vocabulary by iteratively merging frequent characters into subwords and frequent subwords into words. For example, BPE starts with adding all individual single characters to its vocabulary (“a,” “b,” etc.). In the next stage, it merges character combinations that frequently occur together into subwords. For example,
“d” and “e” may be merged into the subword “de,” which is common in many English words
like "define," "depend," "made," and "hidden." The merges are determined by a frequency cutoff. 

<div align="center">
<img src="/img/bpeExample.png" alt="BPE Example" width="400"/>
</div>


## Resources 

The resources used for this project are below. 

1. Raschka, Sebastian. Build a Large Language Model. Manning Publications, 2024. 

