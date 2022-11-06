import pandas as pd
import numpy as np

for i in range(0, 4):
    PATH = '/Users/sabin/Documents/Project/AIDA/Trend-analysis/visual_data/NSI/'
    if i == 0:
        PATH += '임대차3법nsi.csv'
    elif i == 1:
        PATH += '중대재해처벌법nsi.csv'
    elif i == 2:
        PATH += '차별금지법nsi.csv'
    elif i == 3:
        PATH += '탄소중립nsi.csv'
        
    data = pd.read_csv(PATH, encoding='UTF8')
    data['Negative_percent'] = data['Negative'] / (data['Negative']+data['Positive'])
    data['Positive_percent'] = data['Positive'] / (data['Negative']+data['Positive'])
    
    PATH = PATH[:-3] + 'xlsx'
    data.to_excel(PATH, index=False)