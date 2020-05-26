import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

class Tokenizer:
    def __init__(self, file_name, model_name, vocab_size):
        self.vocab_size = vocab_size
        self.model_name = model_name
        # FIXME: 주석 없애자.
        spm.SentencePieceTrainer.Train(input=file_name, model_prefix=model_name,\
vocab_size=vocab_size, model_type='bpe')
        # pad_id = -1, unk_id = 0, bos_id = 1, eos_id = 2
        # ----- Model Training Finish -----
        # get model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_name+".model")
        self.sp.SetEncodeExtraOptions("bos:eos")

    def tokenize(self, str):
        return self.sp.Encode(str)

    def detokenize(self, list):
        return self.sp.Decode(list)

class Data:
    def __init__(self, file_name, model_name, vocab_size=37000):
        self.data = []
        self.max_seq_len = 0
        # self.data_len = []

        self.T = Tokenizer(file_name, model_name, vocab_size)

        with open(file_name, "r") as f:
            for line in f.readlines():
                l = self.T.tokenize(line)
                self.data.append(torch.tensor(l, dtype=torch.long).cuda())

                # self.data_len.append(len(l))
                if self.max_seq_len < len(l):
                    self.max_seq_len = len(l)
                #FIXME:
                # if len(self.data) > 1000:
                #     break

        # self.data = pad_sequence(self.data, batch_first=True)
        # self.data_len = torch.tensor(self.data_len)
        
        # FIXME: sort by seq_len in descending order
        # self.seq_lengths, perm_idx = self.data_len.sort(0, descending=True)
        # self.data = self.data[perm_idx]
        