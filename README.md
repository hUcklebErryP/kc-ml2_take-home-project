# kc-ml2_take-home-project
KAIST EE lee hae in (이해인)
**Attention Is All You Need.**
~this is code~
_this is tilt._
## 1. introduction
seqential modeling이나 transduction problem (language modeling, machine translation 등)을 푸는데 RNN, LSTM, GRN 등이 많이 쓰였다.
많은 발전이 있었지만 이들은 본질적으로 각 sequence로 hidden state를 계산하는 방법은 병렬화가 불가능하므로 sequence 길이가 길다면 memory limit으로 인해 example들을 배치하는 것이 중요해진다. 
많은 논문들이 이를 해결했지만 sequential computation은 여전히 남아있다.

이 논문은 transformer라는 model architecture를 제안하는데, 이는 recurrence를 없애고 input과 output의 dependency를 알기 위한 attention mechanism을 이용한다.
Transformer는 매우 병렬화되어 짧은 시간 안에 좋은 성능을 선보인다.

## 2. background

## 3. Model Architecture
x -> (encoder) -> z -> (decoder) -> y
### 3.1 Encoder and Decoder Stacks
Encoder, Decoder에 대한 설명
stack layer = 6
### 3.2 Attention
#### 3.2.1 Scaled Dot-Product Attention
#### 3.2.2 Multi-Head Attention
#### 3.2.3 Application of Attention in out Model
Transformer는 multi-head attention을 3가지 방법으로 사용
1. In "encoder-decoder attention" layers
  Q = previous decoder layer output, K, V = encoder output
  decoder의 모든 위치에서 각각 input sequence의 모든 위치와의 attention을 따짐.
2. In encoder, self attention
  Q, K, V = previous encoder output
  encoder의 모든 위치에서 각각 previous encoder 의 모든 위치와의 attention을 따짐.
2. In decoder, self attention
  왼쪽 정보를 지켜야 함. auto-regressive property를 보존하기 위해.
  이를 masking으로 구현.

### 3.3 Position-wise Feed-Forward Networks
### 3.4 Embeddings and Softmax
### 3.5 Positional Encoding

## 4. Why Self-Attention
## 5. Training
### 5.1 Training Data and Batching
Train Data : WMT 2014 English-German dataset.
Sentences는 byte-pair encoding (shared source-target vocabulary of about 37000 tokens를 가짐) 을 통해 인코딩됨. 
### 5.2 Hardware and Schedule
논문에서는 8 NVIDIA P100 GPUs으로 훈련.
각 training step은 0.4s 걸림.
총 100,000 steps (12 hours) 걸림.
### 5.3 Optimizer
베타1 = 0.9, 베타2 = 0.98, 입실론 = 10^-9
learning rate = ...생략...
warmup_steps = 4000.
### 5.4 Regularization
1. Residual Dropout
  각 sub-layer의 output에 drouput을 적용. (add-norm 되기 전)
  encoder, decoder 안에 sums of embeddings, positional encodings에도 적용.
  for base, P_drop = 0.1

2. Label Smoothing
  훈련 중, 입실론_ls = 0.1을 적용
