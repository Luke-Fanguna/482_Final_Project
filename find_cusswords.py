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

labels = pd.read_csv('outputs.csv').fillna({'reason':""})
_labels = labels[labels['predictions'] == True].head()
scores = []
for l in _labels['text']:
    results = Detoxify('original').predict(l)
    _, score = reason(results)
    scores.append(score)
_labels['score'] = scores
combined_df = pd.merge(labels, _labels, how='left', on=['text']).fillna({'score':0,'reason':''})
combined_df.to_csv('scores.csv',columns=['predictions_x','reason_x','text','score'], index=False)



