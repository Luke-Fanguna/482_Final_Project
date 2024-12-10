from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

MODEL = f"cardiffnlp/xlm-twitter-politics-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/xlm-twitter-politics-sentiment')
analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True, top_k=None)

from textsum.summarize import Summarizer

def get_sentiment(text):
  sentiment = analyzer(text)
  return sentiment[0]


def get_avg_sentiment(sentiments):
  sentiment_score = 0
  for sentiment in sentiments:
    score = 0
    for label_with_score in sentiment:
      if label_with_score['label'] == 'Positive':
        score += 1 * label_with_score['score']
      elif label_with_score['label'] == 'Negative':
        score -= 1 * label_with_score['score']
    sentiment_score += score
  return sentiment_score / len(sentiments)


def create_speakers(utterances, people_map):
  speakers = {}

  for utterance, pid in utterances:
    name = people_map[pid][1] + " " + people_map[pid][0]

    if name not in speakers:
      speakers[name] = {'utterances': [], 'sentiment': []}
    speakers[name]['utterances'].append(utterance)
    speakers[name]['sentiment'].append(get_sentiment(utterance))

  for speaker in speakers:
    speakers[speaker]['avg_sentiment'] = get_avg_sentiment(speakers[speaker]['sentiment'])

  return speakers


def get_utterances_summary_large_book(speakers, speaker):
  utterances = speakers[speaker]['utterances']

  summarizer = Summarizer(model_name_or_path="pszemraj/led-large-book-summary", token_batch_length=4096,)
  return summarizer.summarize_string(speaker + ':' + ' '.join(utterances))


def get_utterances_summary_bart(speakers, speaker):
  utterances = speakers[speaker]['utterances']

  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  summary = summarizer(' '.join(utterances), do_sample=False)[0]['summary_text']
  return summary


def find_min_max_speaker_sentiment(speakers):
  min_speaker, max_speaker = None, None
  min_avg_sentiment, max_avg_sentiment = float('inf'), float('-inf')

  # remove speakers who only speak once
  filtered_speakers = {}
  for speaker in speakers:
    if len(speakers[speaker]['utterances']) > 1:
      filtered_speakers[speaker] = speakers[speaker]

  for speaker, speaker_data in filtered_speakers.items():
    avg_sentiment = speaker_data['avg_sentiment']
    
    if avg_sentiment < min_avg_sentiment:
      min_avg_sentiment = avg_sentiment
      min_speaker = speaker
    
    if avg_sentiment > max_avg_sentiment:
      max_avg_sentiment = avg_sentiment
      max_speaker = speaker

  return (min_speaker, min_avg_sentiment), (max_speaker, max_avg_sentiment)


def find_min_max_speaker_utterances(speakers):
  min_utterances_speakers, max_utterances_speakers = [], []
  min_utterances, max_utterances = float('inf'), float('-inf')

  for speaker, speaker_data in speakers.items():
    utterances = speaker_data['utterances']
    num_utterances = len(utterances)
    
    if num_utterances < min_utterances:
      min_utterances = num_utterances
      min_utterances_speakers = [speaker]
    elif num_utterances == min_utterances:
      min_utterances_speakers.append(speaker)

    if num_utterances > max_utterances:
      max_utterances = num_utterances
      max_utterances_speakers = [speaker]
    elif num_utterances == max_utterances:
      max_utterances_speakers.append(speaker)

  return (min_utterances_speakers, min_utterances), (max_utterances_speakers, max_utterances)

def calculate_stats(speakers, print_output=False):
  total_times_spoken = sum([len(v['utterances']) for v in speakers.values()])

  if print_output:
    print(f'Num speakers: {len(speakers)}')
    print(f'Num times spoken: {total_times_spoken}')
    print('\n')

  for speaker in speakers:
    avg_sentiment = speakers[speaker]['avg_sentiment']
    num_times_spoken = len(speakers[speaker]['utterances'])
    speaking_proportion = num_times_spoken / total_times_spoken
    speakers[speaker]['speaking_proportion'] = speaking_proportion
    speakers[speaker]['parsed_words'] = [simple_preprocess(utterance) for utterance in speakers[speaker]['utterances']]
    speakers[speaker]['num_words_spoken'] = sum([len(parsed_word) for parsed_word in speakers[speaker]['parsed_words']])
    
    if print_output:
      print(f'{speaker} spoke {num_times_spoken} / {total_times_spoken} times')
      print(f'Avg sentiment of speaker {speaker}: {avg_sentiment}')
      print('\n')

  total_num_words_spoken = sum([speakers[speaker]['num_words_spoken'] for speaker in speakers])
  print(f'Total words spoken: {total_num_words_spoken}')

  for speaker in speakers:
    speakers[speaker]['words_spoken_proportion'] = speakers[speaker]['num_words_spoken'] / total_num_words_spoken


def get_speaker_sentiment_phenoms(speakers):
  phenoms = []

  (min_speaker, _), (max_speaker, _) = find_min_max_speaker_sentiment(speakers)
  phenoms.append(f"{min_speaker} was the most negative speaker throughout the course of the hearing. Here is a summary of what they said: {get_utterances_summary_mistral(speakers, min_speaker)}")
  phenoms.append(f"{max_speaker} was the most positive speaker throughout the course of the hearing. Here is a summary of what they said: {get_utterances_summary_mistral(speakers, max_speaker)}")

  (min_utterances_speakers, min_utterances), (max_utterances_speakers, max_utterances) = find_min_max_speaker_utterances(speakers)
  phenoms.append(f"{join_multiple(min_utterances_speakers)} spoke the least during the hearing, speaking only {min_utterances} times.")
  phenoms.append(f"{join_multiple(max_utterances_speakers)} spoke the most during the hearing, speaking {max_utterances} times.")

  return phenoms

import nltk
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import re
import logging

logging.getLogger('gensim').setLevel(logging.ERROR)

nltk.download('stopwords', quiet=True)

from huggingface_hub import InferenceClient
client = InferenceClient(api_key="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")

def get_utterances_summary_mistral(speakers, speaker):
  utterances = speakers[speaker]['utterances']
  print(utterances)

  messages = [
    {
      "role": "user",
      "content": f"Provide a complete and concise one or two sentence analysis of {speaker}'s opinion based only their utterances: {' '.join(utterances)}."
    }
  ]

  completion = client.chat.completions.create(model="mistralai/Mistral-Nemo-Instruct-2407", messages=messages, max_tokens=500)
  category = completion.choices[0].message.content

  return category


def preprocess(text, stop_words):
  result = []
  for token in simple_preprocess(text, deacc=True):
    if token not in stop_words and len(token) > 3:
      result.append(token)
  return result


def get_topic_lists(utterances, people_map):
  texts = [text for text, _ in utterances]

  names = set([last for last, _ in people_map.values()] + [first for _, first in people_map.values()])

  stop_words = set(stopwords.words(['english']))

  processed_documents = [preprocess(t, stop_words | names) for t in texts]

  dictionary = corpora.Dictionary(processed_documents)
  corpus = [dictionary.doc2bow(doc) for doc in processed_documents]
  
  lda_model = LdaModel(corpus, id2word=dictionary)
  topics = lda_model.print_topics(num_topics=5, num_words=15)

  topics_ls = []
  for topic in topics:
    words = topic[1].split("+")
    topic_words = [word.split("*")[1].replace('"', '').strip() for word in words]
    topics_ls.append(topic_words)
  return topics_ls

from huggingface_hub import InferenceClient
client = InferenceClient(api_key="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")

def split_camel_case(text):
  words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', text)
  return words


def get_political_category_from_topic_lists(topic_list):
  messages = [{
    "role": "user",
    "content": f"Generate a political category for these words: {topic_list}. Categories should be relevant and specific political issues. Respond only with a single category."
  }]

  completion = client.chat.completions.create(model="mistralai/Mistral-Nemo-Instruct-2407", messages=messages, max_tokens=500)
  category = completion.choices[0].message.content.replace('"', "").replace('_', ' ').replace("*", "").replace('.', '')
  category = re.sub(r'[^a-zA-Z0-9\s]', '', category)
  return category


def get_topics_discussed(utterances, people_map):
  topic_list = get_topic_lists(utterances, people_map)

  categories = set()
  for topic in topic_list:
    categories.add(get_political_category_from_topic_lists(topic))

  return f'Political topics discussed during the hearing included {join_multiple(list(categories))}.'


def detect_dominance(speakers):
  max_spoken_words_proportion_speaker = max(speakers, key=lambda speaker: speakers[speaker]['words_spoken_proportion'])
  proportion = speakers[max_spoken_words_proportion_speaker]['words_spoken_proportion']
  if proportion > 0.4:
    return f'{max_spoken_words_proportion_speaker} dominated the discussion, participating in {proportion * 100:.2f}% of all discussion.'

  return None


def join_multiple(items):
  if len(items) > 1:
    result = ', '.join(items[:-1]) + ', and ' + items[-1]
  else:
    result = items[0] if items else ''

  return result


def get_phenoms(utterances, people_map):
  speakers = create_speakers(utterances, people_map)
  calculate_stats(speakers, print_output=True)

  all_phenoms = []
  sentiment_phenoms = get_speaker_sentiment_phenoms(speakers)

  topics_phenom = get_topics_discussed(utterances, people_map)
  dominance_phenom = detect_dominance(speakers)

  all_phenoms.extend(sentiment_phenoms)
  all_phenoms.append(topics_phenom)
  if dominance_phenom:
    all_phenoms.append(dominance_phenom)

  return all_phenoms
