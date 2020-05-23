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
