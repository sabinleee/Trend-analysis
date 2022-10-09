from glob import glob
import logging
import re
import logging
import json

import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import torch

class keyWord():
    def __init__(self, path='/Users/sabin/Documents/Project/AIDA/Trend-analysis/dataset/임대차3법(54,752건)',
                 sheet='뉴스', model_name='jhgan/ko-sbert-multitask', top_n=5, nr_candidates=10):
        logging.info('Loading data & model...')
        logging.info('It takes a few minutes...')
        
        self.model = SentenceTransformer(model_name)
        self.top_n = top_n
        self.nr_candidates = nr_candidates
        self.okt = Okt()
        self.path = path
        self.sheet = sheet
        self.df = None
        self.preprocessing()
        
    def preprocessing(self):
        self.df = self.load_all_data()
        self.df.columns = self.df.iloc[0]
        self.df.drop(0, inplace=True, axis=0)
        self.df = self.df.dropna(axis=0, how='any')
        self.df['내용'] = self.df['내용'].map(self.text_cleaning)
        self.df = self.df.sort_values(by='작성일')
        
    def load_data(self, file, sheet_name):
        return pd.read_excel(file, sheet_name=sheet_name)

    def load_all_data(self):
        files = glob(self.path+'/*.xlsx')
        df = pd.DataFrame()
        for file in files:
            df = df.append(self.load_data(file, self.sheet))
        return df
        
    def get_keyword(self, text):
        doc_embedding = self.model.encode(text)
        candidates = self.okt.nouns(text)
        candidate_embeddings = self.model.encode(candidates)
        keywords = self.max_sum_sim(doc_embedding, candidate_embeddings, 
                               candidates, self.top_n, self.nr_candidates)
        return keywords
        
    def text_cleaning(self, x):
        mail_del = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z-.]+)", "", str(x))
        meta_del = re.sub("[\r\n\xa0]", "", str(mail_del))
        name_del = re.sub("(\.\s+[ㄱ-ㅎ가-힣]+\s[기]+[자]+)", "", str(meta_del))
        clean_text = re.sub("[^\w\s^.]", " ", name_del)
        
        return clean_text

    def max_sum_sim(self, doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        distances_candidates = cosine_similarity(candidate_embeddings, 
                                                candidate_embeddings)
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [words[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in words]
    
    
def main():
    keyword = keyWord() # path='' 지정해서 사용, 하위 이슈 폴더를 mount
    
    keywords = []
    np.random.seed(42)
    for i in tqdm(np.random.choice(keyword.df.index, 10000)):
        tokenized_doc = keyword.okt.pos(keyword.df.iloc[i]['내용'], norm=True, stem=True)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

        n_gram_range = (1,2)

        count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
        candidates = count.get_feature_names_out()

        candidate_embeddings = keyword.model.encode(candidates, convert_to_tensor=True)
        doc_embedding = keyword.model.encode([tokenized_nouns], convert_to_tensor=True)
        
        keywords.append({keyword.df.iloc[i]['작성일']:keyword.max_sum_sim(doc_embedding.cpu(), candidate_embeddings.cpu(), candidates, top_n=5, nr_candidates=10)})

    with open('/content/drive/MyDrive/aida/keywords.json', 'w', encoding='utf-8') as f:
        json.dump(keywords, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    main()