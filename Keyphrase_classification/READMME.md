# Keyphrase Classfication

    추출한 키워드를 세계(사회), 경제 , 기술과학, 정치 4가지 카테고리로 분류

## Dataset

[기술과학 분야 한영 번역 병렬 말뭉치 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71266)
    
    train set: 120만 데이터 중 일부분 사용
    valid set: 4만

## model

    hugginface : klue/bert-base fine-tunning 


### 추가 사항

15가지의 소분류로 분류 진행

    