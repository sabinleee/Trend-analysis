from google.colab import drive
from tqdm import tqdm,tqdm_notebook
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader

def text_cleaning(x):
    mail_del = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z-.]+)", "", str(x))
    meta_del = re.sub("[\r\n\xa0]", "", str(mail_del))
    name_del = re.sub("(\.\s+[ㄱ-ㅎ가-힣]+\s[기]+[자]+)", "", str(meta_del))
    clean_text = re.sub("[^\w\s^.]", " ", name_del)
    return clean_text


def get_data(filepath):
    drive.mount('/content/drive')

    df = pd.read_excel(filepath)
    df.columns = df.loc[0]
    df.drop(index=0, inplace=True)
    df = df.reset_index(drop=True)

    df['내용'] = df['내용'].map(text_cleaning)
    df = df.sort_values(by=['작성일'])
    return df


class NewsDataset(Dataset):

    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = ' '.join(text.values)
        self.sentences = self.data.split('. ')
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            str(self.sentences[index]),
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'text': str(self.sentences[index])
        }


def model_eval(model, tokenizer, device,params, df):
    dates = df['작성일'].unique()
    pos_neg = [[0, 0, 0] for _ in range(len(dates))]

    model.eval()
    for i, date in enumerate(tqdm_notebook(dates)):

        NewsInput = NewsDataset(df.loc[df['작성일'] == date, '내용'], tokenizer, 128)
        News_loader = DataLoader(NewsInput, **params)

        positive = 0
        negative = 0
        for _, data in enumerate(News_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            text = data['text']
            outputs = model(ids, mask, token_type_ids)

            preds = outputs[0].argmax(dim=1)

            positive += (preds == 1).sum()
            negative += (preds == 2).sum()

        pos_neg[i][0] = date
        pos_neg[i][1] = positive.item()
        pos_neg[i][2] = negative.item()

    df_pos_neg = pd.DataFrame(pos_neg, columns=['date', 'Positive', 'Negative'])

    result = df_pos_neg.rolling(window=7, min_periods=1, on='date').sum()
    result['X_t'] = (result['Positive'] - result['Negative']) / (result['Positive'] + result['Negative'])
    result = result.set_index('date')

    return result


def NSI(date, result):
    X_t = result.loc[date, 'X_t']
    X_hat = result.loc[:date, 'X_t'].mean()
    S = result.loc[:date, 'X_t'].std()

    return float('{:.3f}'.format((X_t - X_hat) / S * 10 + 100))


if __name__ == 'main':

    filepath = '/content/drive/MyDrive/data/SNS수집데이터셋/임대차3법(54,752건)/임대차3법_2021년1월~2022년6월.xlsx'

    df = get_data(filepath)
    dates = df['작성일'].unique()
    News_params = {'batch_size': 16,
                   'shuffle': False,
                   'num_workers': 0
                   }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ''
    tokenizer = ''
    result = model_eval(model, tokenizer,device, News_params, df)

    NSI('2021/03/15', result)
