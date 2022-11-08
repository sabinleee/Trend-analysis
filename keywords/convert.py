import json
import pandas as pd

for i in range(1, 5):
    path = '/Users/sabin/Documents/Project/AIDA/Trend-analysis/keywords/'
    if i == 1:
        path += "rent.json"
    elif i == 2:
        path += "accident.json"
    elif i == 3:
        path += "discrimination.json"
    elif i == 4:
        path += "neutral.json"
    
    content_path = '/Users/sabin/Documents/Project/AIDA/Trend-analysis/keywords/'
    if i == 1:
        content_path += "rent_multi_content.json"
    elif i == 2:
        content_path += "accident_multi_content.json"
    elif i == 3:
        content_path += "discrimination_multi_content.json"
    elif i == 4:
        content_path += "neutral_multi_content.json"
        
    with open(path, 'r') as file:
        keywords = json.load(file)
        
    with open(content_path, 'r') as file:
        keywords_multi = json.load(file)
        
    keywords_multi = [keywords_multi[i:i+5] for i in range(0, len(keywords_multi), 5)]

    date = []
    words = []
    for keyword in keywords:
        date.append(list(keyword.keys())[0])
        words.append(list(keyword.values())[0])
        
    df_keywords = pd.DataFrame({'date':date, 'words':words})

    for j in range(5):
        df_keywords[f'keyword{j}'] = df_keywords['words'].apply(lambda x: x[j])
        df_keywords[f'keyword{j}_content'] = [keyword[j][1] for keyword in keywords_multi]    
        
    df_keywords.sort_values(by='date', inplace=True)
    
    save_path = '/Users/sabin/Documents/Project/AIDA/Trend-analysis/keywords/'
    if i == 1:
        save_path += "rent.xlsx"
    elif i == 2:
        save_path += "accident.xlsx"
    elif i == 3:
        save_path += "discrimination.xlsx"
    elif i == 4:
        save_path += "neutral.xlsx"
    
    print(save_path)    
    df_keywords.to_excel(save_path, index=False)