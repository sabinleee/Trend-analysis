# Trend-analysis
뉴스와 소셜 데이터 기반 이슈 분석 및 시각화

<hr>

## 개요
전이학습을 통한 대중적 관심분야 분석 및 통계 : 뉴스와 소셜 데이터를 중심으로

## training environment
- python 3.9
- CUDA 11.2
- pytorch 1.7
- openjdk-8-jdk
- 각 모델 설정값 참고

## directory strcture
```
├── anaconda3
├── dataset
│   ├── chabeul
│   │   ├── chabeul_2021_01_04.xlsx
│   │   ├── *.xlsx
│   ├── chungdae
│   ├── imdeacha
│   ├── tanso
├── model
│   ├── description
│   │   ├── models
│   │   │   ├── chabeul
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model.bin
│   │   │   │   ├── special_tokens_map.json
│   │   │   │   ├── tokenizer_config.json
│   │   │   │   ├── tokenizer.json
│   │   │   │   ├── training_args.bin
│   │   │   ├── chungdae
│   │   │   ├── imdeacha
│   │   │   ├── tanso
│   │   ├── src
│   │   │   ├── example
│   │   │   ├── description.py
│   │   │   ├── train.py
│   │   │   ├── utils.py
│   ├── issuemulticlassification
│   │   ├── dataset
│   │   │   ├── issue_test_set.csv
│   │   │   ├── issue_train_set.csv
│   │   ├── models
│   │   │   ├── issue_classification.pt
│   │   ├── src
│   │   │   ├── evaluation.py
│   │   │   ├── utils.py
│   ├── keyword
│   ├── nsi
│   ├── sentimental
├── result
│   ├── 임대차3법
│   │   ├── keywords_뉴스.xlsx
│   │   ├── keywords_블로그.xlsx
│   │   ├── keywords_커뮤니티.xlsx
│   │   ├── keywords_트위터.xlsx
│   │   ├── nsi_뉴스.xlsx
│   │   ├── sentimental_binary_블로그.xlsx
│   │   ├── sentimental_binary_커뮤니티.xlsx
│   │   ├── sentimental_binary_트위터.xlsx
│   │   ├── wordcloud_뉴스.png
│   │   ├── wordcloud_블로그.png
│   │   ├── wordcloud_커뮤니티.png
│   │   ├── wordcloud_트위터.png
│   ├── 중대재해처벌법
│   ├── 차별금지법
│   ├── 탄소중립
│   ├── 발표자료.pdf
└── README.md
```

## model
1. description
   * attention 기반 transformer text generation model
   * KoGPT-2를 이용한 문장 독해 및 생성 모델
2. issue classification
   * Fine tuning in terms of specific task on Pre-training of Deep Bidirectional Transformers 
   * KoBERT 모델을 이용한 이슈 분류 모델
3. keyword extraction
   * Attention 기반 Transformer sentence embedding model
   * n-gram 및 max-sum-similarity를 이용한 키워드 추출 모델
4. nsi
   * KoBERT 기반 경제, 사회 흐름 및 이슈 해석 모델
5. sentimental classification
   * 트위터,커뮤니티,블로그 데이터 감성분석 모델

## how to use
1. description
   * 실행 : python model/description/description.py
   * predict : 실행 후 원하는 항목 수동 입력 
     * **원하는 항목을 물음 형태로 제시**
     * ex) 임대차 3법이란?
     * 생성 모델이므로 매 실행마다 유사한 의미를 가지나 결과가 다를 수 있음
2. issue classification
   * 실행 : python model/issuemulticlassification/evaluation.py
   * predict : interative한 입력을 하고싶었으나, 적절한 데이터셋을 구하지 못해 csv 파일로 입력
   * 결과물 : result 폴더의 keyword.xlsx 참고
3. keyword extraction
   * 실행 : python model/keyword/main.py
   * predict : interative한 입력을 하고싶었으나, 적절한 데이터셋을 구하지 못해 csv 파일로 입력
   * 결과물 : result 폴더의 keyword.xlsx 참고
4. nsi
   * 실행 : python model/nsi/main.py
   * predict : 일주일 이상의 데이터가 입력으로 들어와야 유의미한 지표 도출가능
   * 결과물 : result 폴더의 nsi_뉴스.xlsx 참고
5. sentimental classification
   * 실행 : python model/sentimental/main.py
   * predict : 직접 입력하여 확인하거나, csv 파일로 입력
   * 결과물 : result 폴더의 sentimental_binary_*.xlsx 참고

## requirements
   * conda 환경 아래 base 변수에 필요한 패키지 설치해두었습니다.

## demo video


## etc
**자세한 사항은 발표자료를 참고해주세요.**
