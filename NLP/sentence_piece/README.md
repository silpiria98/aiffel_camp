🔑 **PRT(Peer Review Template)**

- [O] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
  - [O] 1. SentencePiece를 이용하여 모델을 만들기까지의 과정이 정상적으로 진행되었는가?
          ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/116686a5-b32d-4b56-8529-bd2362fe4532)
          ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/d8089029-215b-4601-902a-fa23879e2875)
          - bpe, unigram 사용한 모델을 모두 정상적으로 학습 완료했습니다. 
  - [O] 2. SentencePiece를 통해 만든 Tokenizer가 자연어처리 모델과 결합하여 동작하는가?
          ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/1d5b0e9a-8c3f-4c12-b67d-798754859533)
          - 지난 DLthon에서 학습했던 모델과 결합해서 잘 동작하는 것을 확인했습니다(이런 효율적인 방법이...) 

  - [O] 3. SentencePiece의 성능을 다각도로 비교분석하였는가?
          ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/b6d8e56c-6b6e-4354-8bcf-0a0c34621c12)
          ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/b9e3cad5-4888-4685-a99b-c9c5e4fdcd07)
          mecab 분석기와 sentence piece bpe 성능과 비교한 내용이 잘 들어있습니다. 

          
- [▲] **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**

  - [▲] 모델 선정 이유
  - [▲] Metrics 선정 이유
  - [▲] Loss 선정 이유
        ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/791f0f4b-0b2b-431b-abee-1478e55333a3)
        주현님께서 딱히 이유는 쓰지 않으셨는데 이 부분은 통상적으로 해당 프로젝트에 효율적으로 작동하는 metric, loss, 옵티마이저이고, DLthon에서 트랜스포머를 응용해서 학습해 미리 좋은 성능을 낸 모델을 사용한거라 빠른 시간안에 잘 선택해서 사용한 거 같습니다.


- [▲] **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**

  - [o] 데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
        ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/63a7e0c4-811f-464a-a62a-1d952b64be34)
        데이터 분할해서 프로젝트 진행하였습니다~~

  - [x] 하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
  - [o] 각 실험을 시각화하여 비교하였나요?
        ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/7af14790-3ed8-4ef5-84fb-5d519a06d9bd)
        verbose = 1로 학습 과정을 시각화해서 잘 기록하였습니다. 
  - [O] 모든 실험 결과가 기록되었나요?

- [x] **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
  - [x] 배운 점
  - [x] 아쉬운 점
  - [x] 느낀 점
  - [x] 어려웠던 점
   ![image](https://github.com/silpiria98/aiffel_camp/assets/35359870/9cfa65e4-6d36-4b76-8d5f-9cd3e635c423)
  - 회고가 잘 적혀 있습니다.

