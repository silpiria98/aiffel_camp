#전처리
# 불용어 처리를 할 경우 stopwords에 단어 리스트를 제공

# sample stopwords
# ['그렇게','아','어떻게', '이렇게', '그렇군요', '있어요',
# '내가', '다', '니','니가','넌','그냥', '너', '왜', '야','진짜',
# '나','좀', '지금', '내', '아니','우리','네','안','그','이','어',
# '그래', '그럼', '아니야', '응', '너가', '제가', '저', '거','뭐',
# '이거','여기', '저는','저도', '전', '어', '나도', '잘', '너무',
# '정말', '나는', '너도', '네가', '넌', '난', '널',]

# 워드 클라우드에서 보기 편하게 일반어 제거

import re

def preprocess_sentence(sentence, stopwords=None):
    # 개행자 삭제
    sentence = re.sub(r'[\n\r]', ' ', sentence)
    
    # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
    # 예를 들어서 "I am a student." => "I am a student ."와 같이
    # student와 온점 사이에 거리를 만듭니다. 
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # (a-z, A-Z,가-힣,0-9, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다.
    sentence = re.sub(r"[^a-zA-Z가-힣0-9\.\?\!,]", " ", sentence)
    sentence = sentence.strip()

    # '키키'와 같이 연속된 키를 제거합니다.
    sentence = re.sub(r'키{2,}', '', sentence)
    sentence = re.sub(r'\b키\b', '', sentence)

    if stopwords:
        words = sentence.split()
        filtered_words = [word for word in words if word not in stopwords]
        sentence = ' '.join(filtered_words)

    return sentence
