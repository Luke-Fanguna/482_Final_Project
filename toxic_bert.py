from detoxify import Detoxify
import pandas as pd

def reason(result):
    val = 0
    label = ""
    for k, v in result.items():
        if val < v:
            label = k
            val = v
    return label, val

df = pd.read_csv('utterance2324.csv').dropna(axis=0, how='any')
for t in df['text']:
    d = pd.read_csv('scores.csv')
    results = Detoxify('original').predict(t)
    prediction = max(results.values()) > 0.1
    label = ""
    val = 0
    if prediction:
        label, val = reason(results)
    new_row = pd.DataFrame({'predictions': [prediction], 'reason' : label, 'text': [t], 'score': val})
    d = pd.concat([d, new_row], ignore_index=True)
    d.to_csv('scores.csv', index=False)
