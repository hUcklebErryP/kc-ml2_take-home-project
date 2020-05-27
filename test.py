import torch
import torch.nn.functional as F
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_idx', type=int, help="index of sentence")
    args = parser.parse_args()
    data_idx = args.data_idx

    # load model and source, target object.
    with open('SRC', 'rb') as f:
        SRC = pickle.load(f)
    with open('TRG', 'rb') as f:
        TRG = pickle.load(f)
    model = torch.load("./transformer.pt")
    model.eval()

    # output of transformer
    outputs = torch.zeros(TRG.max_seq_len).type_as(SRC.data[0]).cuda()
    # start with bos_id
    outputs[0] = 1

    for i in range(1, TRG.max_seq_len):
        out = model(SRC.data[data_idx], outputs[:i])
        out = F.softmax(out, dim=-1)

        val, ix = out[-1:].data.topk(1)

        outputs[i] = ix[0] # largest one

        if ix[0] == 2:
            break

    for i in range(len(outputs)):
        if outputs[i] == 0:
            outputs = outputs[:i]
            break

    result_string = TRG.T.detokenize(outputs.tolist())

    # print("SRC", SRC.data[data_idx])
    # print("TRG", TRG.data[data_idx])
    # print("translated :", outputs)

    print("source sentence :", SRC.T.detokenize(SRC.data[data_idx].tolist()))
    print("target sentence :", TRG.T.detokenize(TRG.data[data_idx].tolist()))
    print("translated sentence :", result_string)

if __name__=="__main__":
    main()