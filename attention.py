import torch
import torch.nn.functional as F

import math
# import pandas as pd

# dtype = torch.float
# d_model = 512
# d_k = 64
# d_v = 64
# h = 8
device = torch.device("cuda:0")

def scal_dot_attention(Q, K, V, d_k=64, mask=None):
    scores = torch.matmul(Q, K.t() / math.sqrt(d_k)) # size (max_len, max_len)
    # TODO: Mask : Optional FIXME: transpose(-2,-1)
    scores = F.softmax(scores, dim=-1) # FIXME: softmax
    return torch.matmul(scores, V)
    # CHECK: dim=-1?`

class Attention(torch.nn.Module):
    def __init__(self, d_model, h, max_seq_len):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        # self.concat_matrix = torch.zeros(max_seq_len, d_model).to(device)

        self.W_Qs = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d_model, self.d_k)) for i in range(h)])
        self.W_Ks = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d_model, self.d_k)) for i in range(h)])
        self.W_Vs = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d_model, self.d_k)) for i in range(h)])
        self.W_O = torch.nn.Parameter(torch.randn(d_model, d_model))
        
        # self.W_Qs = []
        # self.W_Ks = []
        # self.W_Vs = []
        # for _ in range(h):
        #     self.W_Qs.append(torch.nn.Linear(d_model, self.d_k))
        #     self.W_Ks.append(torch.nn.Linear(d_model, self.d_k))
        #     self.W_Vs.append(torch.nn.Linear(d_model, self.d_k))
        # self.W_O = torch.nn.Linear(d_model, d_model)

    
    def forward(self, Q, K, V, mask=None):
        for i in range(self.h):
            head = scal_dot_attention(torch.matmul(Q, self.W_Qs[i]), \
torch.matmul(K, self.W_Ks[i]), torch.matmul(V, self.W_Vs[i]), self.d_k, mask)
            # head = scal_dot_attention(self.W_Qs[i](Q), self.W_Ks[i](K), self.W_Vs[i](V), self.d_k, mask)
            if i == 0:
                concat_matrix = head.clone()
            else:
                concat_matrix = torch.cat((concat_matrix, head), 1)
        output = torch.matmul(concat_matrix, self.W_O)
        # output = self.W_O(concat_matri)
        return output
