# kc-ml2_take-home-project
KAIST EE lee hae in (이해인)
**Attention Is All You Need.**

# Summary of "Attention Is All You Need"

## 1. introduction

Many algorithm like RNN, LSTM, GRN are used to solve sequential modeling or transduction problem (language modeling, machine translation).
Although there was significant progress, there is still sequential computation that cannot be parallelized.

In this paper, they proposed a new model architecture, transformer.
It removes recurrence and it use totally attention mechanism.

Transformer has good parallelism, so it shows good performance in a short time.

## 3. Model Architecture

x -> (encoder) -> z -> (decoder) -> y

(In this page, I will call z as context vector.)

### 3.1 Encoder and Decoder Stacks

Encoder and Decoder are the layer that are consisted of N sublayers.
Each sublayers, there are attention, feed forward network, and Add-Norm.

### 3.2 Attention

In this paper, they declare multi-head attention.
Transformer use multi-head attention in 3 ways.

1. In Encoder, self attention.
Q, K, V = previous encoder output
Q concerns about all position in source sequence.

2. In Decoder, self attention.
Q, K, V = previous decoder output
Q concerns about all position in target sequence.

3. In Decoder, masked attention.
Q = previous Add-Norm output in Decoder
K, V = context vector
To learn next token from previous positions, mask has to be used.

### 3.3 Positional Encoding

This model does not check position by recurrence.
They solved it by adding positional encoding to input.
This makes different between same tokens, different positions.

# Run

## 1. Environment

pytorch 1.5.0
sentencepiece

## 2. Command
### 2.1 Train

Run, `python train.py`

It will tokenize source data and target data.
Then, model will be training.
We can adjust hyperparameters in train.py file.

### 2.2 Test

Run, `python test.py X`, X is nonnegative integer like 0, 1, 2, ...

X indicdates the index of sentece in train data.
This command will show you index X of source sentence, target sentence, and translated sentence.


# Explanation

## 1. Preprocess (in preprocess.py)

I use sentencepiece module to tokenize sentences.

class Tokenizer do tokenizing over sentences.
Its model type is bpe(byte-pair encoding).
Note that bos_id is 1, eos_id is 2

class Data is a class which manages text raw file.
Its member variable data is a list of tensors which have tokens of a sentence in 1d.
(example, [tensor([1,3,2]), tensor([1,4,5,2])])
I use list type because I have to handle various length of sentences.


## 2. Train (in train.py)

I use IWSLT'15 English-Vietnamese data. (133k sentences)
`train.en`, `train.vi` are raw english, vietnamese data.
`train_s.en`, `train_s.vi` are top 1k sentences of raw data, respectively.

From preprocessing, SRC, TRG stores its data.
(In this case, SRC is english training set and TRG is vietnamese training set.)

`model` is a transformer model which will be trained.
It will be trained in cuda:0.

Optimizer : Adam
loss function : cross_entropy

## 3. Modules (in transformer.py, attention.py)

### 3.1 Attention (in attention.py)

class Attention is a custom multi-head attention module.

For input Q, K, V tensor, do linear process using W_Qs, W_Ks, W_Vs.
Do scalar dot product attention function to each submatrix, and concat them.
Last, do linear process using W_O.

`scal_dot_attention()` is a function that proceed scalar dot product attention.
mask is for decoder, it will mask Q tensor's part. (assign -1e9)

### 3.2 class Embedding (in transformer.py)

class Embedding is a layer that embed input sequence.
Input sequence is a tensor which is 1d matrix, stores input sequence tokens.
(Example : tensor([1, 244, 35, 2]))
check, 1 is bos_id and 2 is eos_id.

This layer will embed the tokens to d_model-dimension vector.

### 3.3 Positional_Encoding (in transformer.py)

class Positional_Encoding is layer that proceed positional encoding.
I use pe_matrix which size is (max_seq_len x d_model), and has PE values.
When input comes, it add PE in right size (seq_len x d_model).
check, pe_matrix is buffer which is not changed during training.

### 3.4 FeedForwardNN (in transformer.py)

class FeedForwardNN is a layer that proceed feed forward network.
when input (size : seq_len x d_model) comes, it goes to d_ff.
After relu activation, it goes to d_model.

### 3.5 Encoder, Decoder (in transformer.py)

class Encoder, Decoder are sublayer of total Encoder and Decoder.
check, total Encoder and Decoder are consists of N sublayers.
check, in Decoder, masked multi head attention is used.

### 3.6 Encoders, Decoders (in transformer.py)

class Encoders, Decoders are total modules.
Encoders will get input from SRC, and throw output as context_vector.
Decoders will get input from context_vector and input form TRG, and throw output.

### 3.7 Transformer (in transformer.py)

class Transformer is a model which is all in one.

## 4. Result (in test.py)

Sadly, my model does not work properly.

I can check it in test.py.

Load model, and SRC, TRG object.
For given number I (index of sentence in train data), it will print source sentence, target sentence, and translated sentence by model.

There are big difference between them.

However, during training, I can check the loss average are pushed down around 6.

## 5. Code Reference

https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
https://github.com/google/sentencepiece
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
https://pytorch.org/docs/stable/nn.html
