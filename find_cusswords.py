import pandas as pd

utterance = pd.read_csv('utterances.csv').dropna(axis=0, how='any')
utterance['text'] = utterance['text'].str.lower()
print(utterance[utterance['text'].str.contains('shit')])
print(utterance['text'][393174])