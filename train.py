import time
import torch
import torch.nn.functional as F
import preprocess
import transformer
import pickle

vocab_size = 5534
d_model = 128
h = 8
N = 3
dp = 0.1
d_ff = 512
epochs = 10

SRC = preprocess.Data("./train_s.en", "src", vocab_size)
TRG = preprocess.Data("./train_s.vi", "trg", vocab_size)

model = transformer.Transformer(vocab_size, vocab_size, d_model, h, \
    SRC.max_seq_len, TRG.max_seq_len, N, dp, d_ff)
# model = torch.nn.DataParallel(model)
model = model.to('cuda:0')

# for p in model.parameters():
#     if p.dim() > 1:
#         torch.nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()

start = time.time()
temp = start

total_loss = 0
for epoch in range(epochs):
    
    for i in range(len(SRC.data)):

        preds = model(SRC.data[i], TRG.data[i][:-1])

        optim.zero_grad()
        
        loss = F.cross_entropy(preds, TRG.data[i][1:])

        loss.backward()
        optim.step()

        total_loss += loss
        if (i + 1) % 100 == 0:
                loss_avg = total_loss / 100
                print("time : %dm %ds\tepoch : %d\titer : %d\tloss_avg : %.3f\t\
    %ds per 100 iters" % ((time.time() - start) // 60, (time.time() - start) % 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp))
                total_loss = 0
                temp = time.time()

torch.save(model, "./transformer.pt")
with open('SRC', 'wb') as f:
    pickle.dump(SRC, f)

with open('TRG', 'wb') as f:
    pickle.dump(TRG, f)
