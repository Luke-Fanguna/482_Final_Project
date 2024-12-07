from detoxify import Detoxify
import pandas as pd

df = pd.read_csv('utterance2324.csv').dropna(axis=0, how='any')
output = []
for t in df['text']:
    results = Detoxify('original').predict(t)
    if max(results.values()) > .04:
        output.append(True)
    else:
        output.append(False)
df['LABEL'] = output
df.to_csv('labelled_utterances.csv', index=False)
# results = Detoxify('original').predict("next slide, please. you can skip the video. it doesn't matter. it's just basically anderson calling in preparation for this. it's okay. it's basically anderson calling me a dipshit because this is what my students do for fun. so you've already talked about non consensual sexual imagery. this stuff is being used. you have to understand that this technology is not hard to use. for most people, it's just a matter of going to a website.")
# print(results)
# print((df>.04).any())