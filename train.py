import time
import torch
import torch.nn.functional as F
import preprocess
import transformer
import pickle

vocab_size = 37000
d_model = 512
h = 8
N = 1
dp = 0.1
d_ff = 2048
epochs = 3

SRC = preprocess.Data("./train.en", "src", vocab_size)
TRG = preprocess.Data("./train.vi", "trg", vocab_size)


# SRC = torch.tensor([[1,2,3,4],[2,3,4,5],[3,4,5,6]])

model = transformer.Transformer(vocab_size, vocab_size, d_model, h, \
    SRC.max_seq_len, TRG.max_seq_len, N, dp, d_ff)
# model = torch.nn.DataParallel(model)
model = model.to('cuda:0')

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p) #CHECK:

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def train_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start

    total_loss = 0
    for epoch in range(epochs):
        
        for i in range(len(SRC.data)):
        #     if i > 1000:
        #         break

            preds = model(SRC.data[i], TRG.data[i][:-1])
            
            optim.zero_grad()

            loss = F.cross_entropy(preds, TRG.data[i][1:]) # FIXME:

            loss.backward()
            optim.step()

            total_loss += loss
            if (i + 1) % print_every == 0:
                    loss_avg = total_loss / print_every
                    print("time = %dm, epoch %d, iter = %d, loss = %.3f, \
                    %ds per %d iters" % ((time.time() - start) // 60,
                    epoch + 1, i + 1, loss_avg, time.time() - temp,
                    print_every))
                    total_loss = 0
                    temp = time.time()
        
train_model(epochs)

torch.save(model, "./transformer.pt")
with open('SRC', 'wb') as f:
    pickle.dump(SRC, f)

with open('TRG', 'wb') as f:
    pickle.dump(TRG, f)