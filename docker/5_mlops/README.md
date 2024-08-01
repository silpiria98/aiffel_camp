## MLOps 실습

네이버 영화 리뷰에 대한 긍부정을 판단하는 모델을 훈련하고 fastapi로 호스팅 하는 부분'
모든 mlops 구현은 따로 구현중

```mermaid
flowchart LR
    subgraph mysql
    Database
    end
    Database-->train[Model train]

    subgraph Model[Model Development]
    train-->save[Model Save]
    end

    subgraph FastAPI
    deployment[Model Deployment] -->serve[Model Serving]
    end

    user --request--> serve --response--> user

    subgraph Wandb
    save-->registry[Model Registry]-->deployment
    end

    serve --> Database

    subgraph Thispart
    FastAPI
    mysql
    end
    subgraph MlOps
    Thispart
    Wandb
    Model
    end
```
