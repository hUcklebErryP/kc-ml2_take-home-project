import sentencepiece as spm

class Bpe:
    def __init__(self, _input_file, _vocab_size):
        spm.SentencePieceTrainer.Train(input=_input_file, model_prefix='m',\
            vocab_size=_vocab_size, model_type='bpe')
        # pad_id = -1, unk_id = 0, bos_id = 1, eos_id = 2
        # ----- Model Training Finish -----
        # get model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("m.model")

    def encode(self, str):
        return self.sp.Encode(str)

    def decode(self, list):
        return self.sp.Decode(list)
