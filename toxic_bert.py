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

# df['LABEL'] = output
# df.to_csv('labelled_utterances.csv', index=False)
# results = Detoxify('original').predict("next slide, please. you can skip the video. it doesn't matter. it's just basically anderson calling in preparation for this. it's okay. it's basically anderson calling me a dipshit because this is what my students do for fun. so you've already talked about non consensual sexual imagery. this stuff is being used. you have to understand that this technology is not hard to use. for most people, it's just a matter of going to a website.")
# print(results)
# print((df>.04).any())