🔑 **PRT(Peer Review Template)**

- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
  - [x] 1. 다양한 방법으로 Text Classification 태스크를 성공적으로 구현하였다.
      - [x] 3가지 이상의 모델이 성공적으로 시도됨
        1. ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/34ded351-f48e-4932-8fb0-b227f124bd68)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/f560fd04-f80f-4281-88a8-8848f5d9af7a)
        2. ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/8cd16239-df6c-4789-abcf-3f722cdb4035) ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/cbbd992f-015b-4504-94d4-602b12341838)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/1666217f-9cbb-4447-88a4-55aaad9a4680)
        3. ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/b9affd24-190a-4785-9203-d0c6d5c8f347) ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/c29b8096-37af-4b8e-ac7d-49aa857af239)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/8a018095-7cf4-43a8-8fbf-87f081e84ee0)
  - [x] 2. gensim을 활용하여 자체학습된 혹은 사전학습된 임베딩 레이어를 분석하였다.
      - [x] gensim의 유사단어 찾기를 활용하여 자체학습한 임베딩과 사전학습 임베딩을 비교 분석함. ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/3df766c4-74f8-4e22-8b23-d756f7488557) ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/2808db20-94fc-41f9-bc8d-666b3244aaeb)
  - [x] 3. 한국어 Word2Vec을 활용하여 가시적인 성능향상을 달성했다.
      - [x] 네이버 영화리뷰 데이터 감성분석 정확도를 85% 이상 달성함: ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/1633b2e5-91eb-490e-94f0-327fd73e8316)

- [x] **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
  - 전반적으로 주석보다는 마크다운 형태로 작성해 주셨습니다. 본인의 생각과 결과에 대한 분석을 적었습니다.
  - [x] 모델 선정 이유
  - [x] Metrics 선정 이유
  - [x] Loss 선정 이유

- [x] **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
  - [x] 데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [x]  ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/5aecb035-d421-4e18-93ed-d625e1919a12)
![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/485ba14b-f5a7-4f1f-9f34-877597c35deb)
  - [x] 하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [x] ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/0324a1be-1ee5-4df8-8089-fab8cf62ce59)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/fae7f290-3785-4ca3-9a02-6e90c70b9c22)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/1e4406e6-4ec3-4a13-a854-d21ad1f3629a) ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/bee2b6ec-0e03-40e9-a505-bb76374bb877) ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/6eaef147-ce60-44b9-abb9-dde19dec6171)
  - [x] 각 실험을 시각화하여 비교하였나요?
    - [x] 임베딩 비교: ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/67dded8c-d428-4d5c-b750-6d256922eea0)
![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/45422c55-7024-43b6-bd7d-947defd121bd)
  - [x] 모든 실험 결과가 기록되었나요?
    - [x] ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/f560fd04-f80f-4281-88a8-8848f5d9af7a) ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/cbbd992f-015b-4504-94d4-602b12341838)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/1666217f-9cbb-4447-88a4-55aaad9a4680)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/c29b8096-37af-4b8e-ac7d-49aa857af239)![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/8a018095-7cf4-43a8-8fbf-87f081e84ee0)

- [x] **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    ![image](https://github.com/hojae-m-choi/aiffel_camp/assets/98305832/a3131eda-6627-4e23-9875-b99eddd35b4c)
  - [x] 배운 점
    - lstm 층을 늘리는 것이 성능 향상에 도움을 주기는 하였는데, 계속 늘리는 것은 의미가 없었다.
    - cnn층으로 특성 추출을 하고 lstm층으로 감성분석을 하는 것은 매우매우 성능이 좋지 않았다.
    - 구둣점이나 '을' 같이 제거되지 못한 불용어를 추가제거 하여도 성능 향상에 많은 도움을 주지는 않았다.
  - [x] 아쉬운 점
    - 마지막에 조원분이 lstm레이어 층을 넓히면 성능이 좋아진다고 하셔서 시도해 봐야 할 것 같다.
  - [x] 느낀 점
    - nlp에선 양방향 lstm이 일반 lstm보다 성능이 좋은 걸 체감하였다.
  - [x] 어려웠던 점
    - recurrent dropout을 사용하려고 했는데, cudnn이 작동하지 않아 20분동안 돌리다가 포기
