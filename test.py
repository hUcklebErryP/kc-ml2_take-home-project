import torch
import torch.nn.functional as F
import transformer
import preprocess
import attention
import time
import pickle

with open('SRC', 'rb') as f:
    SRC = pickle.load(f)

with open('TRG', 'rb') as f:
    TRG = pickle.load(f)


model = torch.load("./transformer.pt")
model.eval()

outputs = torch.zeros(TRG.max_seq_len).type_as(SRC.data[0])
outputs[0] = 1

for i in range(1, TRG.max_seq_len):

    out = model(SRC.data[1], outputs[:i])
    out = F.softmax(out, dim=-1)

    val, ix = out[-1:].data.topk(1)

    outputs[i] = ix[0]
    if ix[0] == 2:
        break

for i in range(len(outputs)):
    if outputs[i] == 0:
        outputs = outputs[:i]
        break

result_string = TRG.T.detokenize(outputs.tolist())

print(outputs)
print(result_string)

print("SRC", SRC.data[1])
print("TRG", TRG.data[1])