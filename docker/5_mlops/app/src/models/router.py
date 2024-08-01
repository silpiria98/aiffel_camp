import torch
from kiwipiepy import Kiwi
from models.schemas import SentimentResponse, TextRequest
from fastapi import APIRouter, Depends
from models.vocab import open_json
from models.model import model

router = APIRouter()
kiwi = Kiwi()


stopwords:set[str] = {'도', '는', '다', '의', '가', '이', '은', '한', 
                      '에', '하', '고', '을', '를', '인', '듯', '과',
                      '와', '네', '들', '듯', '지', '임', '게'}
index_to_tag = {0 : '부정', 1 : '긍정'}
file_path = 'src/models/naver_vocab.json'
word_to_index = open_json(file_path)

def preprocessing_with_kiwi(text:str) -> list[str]:
    return [t.form for t in kiwi.tokenize(text)]

@router.post('/predict', response_model=SentimentResponse)
async def predict_sentiment(request:TextRequest)\
    ->SentimentResponse:
        
    # Tokenize the input text
    tokens = preprocessing_with_kiwi(request.sentence) # 토큰화
    tokens = [word for word in tokens if not word in stopwords] # 불용어 제거
    token_indices = [word_to_index.get(token, 1) for token in tokens]

    # Convert tokens to tensor
    input_tensor = torch.tensor([token_indices], dtype=torch.long) # (1, seq_length)

    # Pass the input tensor through the model
    with torch.no_grad():
        logits = model(input_tensor)  # (1, output_dim)

    # Get the predicted class index
    predicted_index = torch.argmax(logits, dim=1)

    # Convert the predicted index to its corresponding tag
    predicted_tag = index_to_tag[predicted_index.item()]

    return SentimentResponse(lab = predicted_tag)