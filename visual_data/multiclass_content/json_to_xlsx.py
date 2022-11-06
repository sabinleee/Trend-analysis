import json
import pandas as pd

for i in range(0, 4):
    PATH = '/Users/sabin/Documents/Project/AIDA/Trend-analysis/visual_data/multiclass_content/'
    if i == 0:
        PATH += 'accident_multi_content.json'
    elif i == 1:
        PATH += 'discrimination_multi_content.json'
    elif i == 2:
        PATH += 'neutral_multi_content.json'
    elif i == 3:
        PATH += 'rent_multi_content.json'
        
    data = pd.read_json('/Users/sabin/Documents/Project/AIDA/Trend-analysis/visual_data/multiclass_content/accident_multi_content.json', encoding='UTF8')
    data.columns = ['keyword', 'content']
    
    PATH = PATH[:-4] + 'xlsx'
    data.to_excel(PATH, index=False)