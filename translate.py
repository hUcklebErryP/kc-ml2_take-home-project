import torch
import preprocess

def translate(model, src, max_seq_len, custom_string=False):
    model.eval()


    
model = torch.load("./MODEL.pt")
model.eval()




def translate(model, src, max_len = 80, custom_string=False):
    
    model.eval()
if custom_sentence == True:
        src = tokenize_en(src)
        sentence=\
        Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok
        in sentence]])).cuda()
src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)
    
    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])
for i in range(1, max_len):    
            
        trg_mask = np.triu(np.ones((1, i, i),
        k=1).astype('uint8')
        trg_mask= Variable(torch.from_numpy(trg_mask) == 0).cuda()
        
        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
        e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)
        
        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break
return ' '.join(
    [FR_TEXT.vocab.itos[ix] for ix in outputs[:i]]
    )