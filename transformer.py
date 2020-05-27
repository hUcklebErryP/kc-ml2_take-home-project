import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import attention
import copy

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        return self.embedding_layer(x)

class Positional_Encoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(Positional_Encoding, self).__init__()        
        self.d_model = d_model
        self.register_buffer("pe_matrix", torch.randn(max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(d_model):
                if i & 1 == 0:
                    # i is even
                    pe = math.sin(pos / (10000 ** (i / d_model)))
                else:
                    # i is odd
                    pe  = math.cos(pos / (1000 ** (i / d_model)))
                self.pe_matrix[pos, i] = pe
    
    def forward(self, x):
        x *= math.sqrt(self.d_model)
        seq_len = x.size(0)
        x += self.pe_matrix[:seq_len, :]
        return x

class FeedForwardNN(torch.nn.Module):
    def __init__(self, d_model, d_ff=512, dp=0.1):
        super(FeedForwardNN, self).__init__()

        self.Linear1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dp)
        self.Linear2 = torch.nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.Linear2(self.dropout(F.relu(self.Linear1(x))))

class Encoder(torch.nn.Module):
    def __init__(self, d_model, h, dp=0.1, d_ff=512):
        super(Encoder, self).__init__()

        self.Norm1 = torch.nn.LayerNorm(d_model)
        self.Norm2 = torch.nn.LayerNorm(d_model)
        self.MultiHeadAttention = attention.Attention(d_model, h)
        self.Dropout1 = torch.nn.Dropout(dp)
        self.Dropout2 = torch.nn.Dropout(dp)
        self.FFNN = FeedForwardNN(d_model, d_ff, dp)

    def forward(self, x):
        x2 = self.Norm1(x + self.Dropout1(self.MultiHeadAttention(x, x, x)))
        return self.Norm2(x2 + self.Dropout2(self.FFNN(x2)))
        
class Decoder(torch.nn.Module):
    def __init__(self, d_model, h, dp=0.1, d_ff=512):
        super(Decoder, self).__init__()
        
        self.MultiHeadAttention1 = attention.Attention(d_model, h)
        self.MultiHeadAttention2 = attention.Attention(d_model, h)
        self.Dropout1 = torch.nn.Dropout(dp)
        self.Dropout2 = torch.nn.Dropout(dp)
        self.Dropout3 = torch.nn.Dropout(dp)
        self.Norm1 = torch.nn.LayerNorm(d_model)
        self.Norm2 = torch.nn.LayerNorm(d_model)
        self.Norm3 = torch.nn.LayerNorm(d_model)
        self.FFNN = FeedForwardNN(d_model, d_ff, dp)

    def forward(self, x, context_vec):
        x1 = self.Norm1(x + self.Dropout1(self.MultiHeadAttention1(x, x, x, mask=True)))
        x2 = self.Norm2(x1 + self.Dropout2(self.MultiHeadAttention2(x1, context_vec, context_vec)))
        return self.Norm3(x2 + self.Dropout3(self.FFNN(x2)))
    
class Encoders(torch.nn.Module):
    def __init__(self, vocab_size, d_model, h, max_seq_len, N=1, dp=0.1, d_ff=512):
        super(Encoders, self).__init__()

        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = Positional_Encoding(d_model, max_seq_len)
        self.encoders = torch.nn.ModuleList([copy.deepcopy(Encoder(d_model, h, dp, d_ff)) for _ in range(N)])

    def forward(self, src):
        temp = self.pe(self.embed(src))
        for i in range(self.N):
            temp = self.encoders[i](temp)
        return temp

class Decoders(torch.nn.Module):
    def __init__(self, vocab_size, d_model, h, max_seq_len, N=1, dp=0.1, d_ff=512):
        super(Decoders, self).__init__()

        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = Positional_Encoding(d_model, max_seq_len)
        self.decoders = torch.nn.ModuleList([copy.deepcopy(Decoder(d_model, h, dp, d_ff)) for _ in range(N)])

    def forward(self, trg, endcoders_output):
        temp = self.pe(self.embed(trg))
        for i in range(self.N):
            temp = self.decoders[i](temp, endcoders_output)
        return temp

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, h, max_seq_len_src, max_seq_len_trg,  N=1,dp=0.1, d_ff=512):
        super(Transformer, self).__init__()
        self.encoders = Encoders(src_vocab_size, d_model, h, max_seq_len_src, N, dp, d_ff)
        self.decoders = Decoders(trg_vocab_size, d_model, h, max_seq_len_trg, N, dp, d_ff)
        self.last_linear = torch.nn.Linear(d_model, trg_vocab_size)

    def forward(self, src_data, trg_data):
        encoders_output = self.encoders(src_data)
        output = self.decoders(trg_data, encoders_output)
        output = self.last_linear(output)
        return output
