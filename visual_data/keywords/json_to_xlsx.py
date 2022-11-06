import json
import pandas as pd

for i in range(0, 4):
    PATH = '/Users/sabin/Documents/Project/AIDA/Trend-analysis/visual_data/keywords/'
    if i == 0:
        PATH += 'accident_keyword.json'
    elif i == 1:
        PATH += 'discrimination_keyword.json'
    elif i == 2:
        PATH += 'neutral_keyword.json'
    elif i == 3:
        PATH += 'rent_keyword.json'
        
    KEYWORD = 5
    with open(PATH, 'r',encoding='UTF8') as file:
        keywords = json.load(file)

    keyword = []
    day = []
    for keyword_dict in keywords:
        keyword.append('/'.join(list(keyword_dict.values())[0]))
        day.append(list(keyword_dict.keys())[0])
    df = pd.DataFrame()
    df = pd.DataFrame({'day':day, 'keyword':keyword}).sort_values(by='day').drop_duplicates().reset_index(drop=True) # 중복된 기사에서 나온 키워드 제거

    for i in range(KEYWORD):
        df[f'keyword{i+1}'] = df['keyword'].map(lambda x:x.split('/')[i])
        
    PATH = PATH[:-4] + 'xlsx'
    df.to_excel(PATH, index=False)