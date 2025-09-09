"""
3.4 Imlementing self-attention with trainable weights

Same logic as simplified_self_attention.py, but this time we are extending the implementation 
of self attention to trainable weights. 

scaled dot-product attention --- the self-attention mechanism used in the orignal transformer architecture. 

Tackling this self-attention mechanism in the two subsections: 
    1. we will code the self-attention mechanism step by step as before. 
    2. we will organize the code into a compact Python class
that can be imported into the LLM architecture.

Note: To better understand why the dot product is so important for machine learning, take a Linear Regression course.
      Or buy a textbook on the subject.

next --> casual attention

"""


