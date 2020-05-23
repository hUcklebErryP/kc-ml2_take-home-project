# kc-ml2_take-home-project
seqential modeling이나 transduction problem (language modeling, machine translation 등)을 푸는데 RNN, LSTM, GRN 등이 많이 쓰였다.
많은 발전이 있었지만 이들은 본질적으로 각 sequence로 hidden state를 계산하는 방법은 병렬화가 불가능하므로 sequence 길이가 길다면 memory limit으로 인해 example들을 배치하는 것이 중요해진다. 
많은 논문들이 이를 해결했지만 sequential computation은 여전히 남아있다.

이 논문은 transformer라는 model architecture를 제안하는데, 이는 recurrence를 없애고 input과 output의 dependency를 알기 위한 attention mechanism을 이용한다.
Transformer는 매우 병렬화되어 짧은 시간 안에 좋은 성능을 선보인다.

