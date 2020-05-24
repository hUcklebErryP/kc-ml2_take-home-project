# import sentencepiece as spm

# if __name__ == "__main__":
#     # 1. Preprocessing

#     ## 1.1 Model Training
#     spm.SentencePieceTrainer.Train(input="./train.en", model_prefix='m',\
#         vocab_size=37000, model_type='bpe')
#                 # pad_id = -1, unk_id = 0, bos_id = 1, eos_id = 2
#     ## 1.2 Get Model
#     sp = spm.SentencePieceProcessor()
#     sp.Load("m.model")

#     print ( sp.Encode("This is nation") )

import bpe

train_en_embed = bpe.Bpe("./train.en", 37000)


