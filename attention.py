import torch
import torch.nn.functional as F
import math

# do scalar dot product attention
def scal_dot_attention(Q, K, V, d_k=64, mask=False):
    scores = torch.matmul(Q, K.t() / math.sqrt(d_k)) # size (max_len, max_len)

    if mask:
        scores = torch.tril(scores) + torch.triu(torch.ones(scores.size(0), scores.size(1)).cuda(), diagonal=1) * -1e9

    scores = F.softmax(scores, dim=-1)
    return torch.matmul(scores, V)

# multi-head attention
class Attention(torch.nn.Module):
    def __init__(self, d_model, h):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_Qs = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d_model, self.d_k)) for i in range(h)])
        self.W_Ks = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d_model, self.d_k)) for i in range(h)])
        self.W_Vs = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d_model, self.d_k)) for i in range(h)])
        self.W_O = torch.nn.Parameter(torch.randn(d_model, d_model))
            
    def forward(self, Q, K, V, mask=False):
        for i in range(self.h):
            head = scal_dot_attention(torch.matmul(Q, self.W_Qs[i]), \
torch.matmul(K, self.W_Ks[i]), torch.matmul(V, self.W_Vs[i]), self.d_k, mask)

            if i == 0:
                concat_matrix = head.clone()
            else:
                concat_matrix = torch.cat((concat_matrix, head), 1)
        output = torch.matmul(concat_matrix, self.W_O)
        return output
