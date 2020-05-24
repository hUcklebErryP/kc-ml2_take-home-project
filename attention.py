import torch
import numpy as np

# import pandas as pd

# class Attention:
#     def __init__(self, Q, K, V):


def Attention(Q, K, V):
    return torch.nn.Softmax(dim=1)( ( Q @ K.t()) / np.sqrt(d_k) ) @ V

def MultiHead(Q, K, V):
    
