import csv

from datetime import datetime, timedelta
#This is a helper function for calculating transcript times
def add_seconds(start_time, seconds_to_add):
  # Parse the start time
  time_obj = datetime.strptime(start_time, '%H:%M:%S')
  # Add the seconds
  new_time = time_obj + timedelta(seconds=seconds_to_add)
  # Format the new time back to string
  return new_time.strftime('%H:%M:%S')

#This is where the unzipped corpus file is stored in Colab file storage
CORPUS_FILE_PATH = 'DH2024_Corpus_Release/'

#We have 4 state sets
VALID_STATES=["CA", "FL", "NY", "TX"]

#Each state set has 9 csv files for each session year
CSV_FILENAMES=['bills','committeeHearings', 'committeeRosters',
               'committees', 'hearings', 'legislature',
               'people','speeches','videos']

#Load data from CSVs into a Python object to reference later
#Input:
#  Required: file_name (type:String) (Ex: speeches, bills, etc)
#  Optional: states (type:List of Strings or None) (Ex: ["CA"], ["FL,TX"])
#     -If not specified (states=None), function returns data for all states
#  Optional: years (type:List of Ints or None) (Ex:[2018], [2015,2016])
#     -If not specified (years=None), function returns data for all valid years
#Output:
#  Payload (Type: Dict) (Ex: {column_headers:['pid','cid','date'], rows:[[0,2,2018],[2,1,2018]]})
def load_content(file_name, states=None, years=None):
  #Only accept valid states, Corpus only contains data on CA, FL, NY, and TX legislations
  if states is not None and not all(item in VALID_STATES for item in states):
    raise Exception("Invalid State Abbv(s), corpus only contains data on CA, FL, NY, and TX")
  #Only accept valid file names from corpus, like speeches, bills, etc.
  if file_name not in CSV_FILENAMES:
    raise Exception("Invalid filename, must be one of the 9 files provide")
  #Only accept years belonging to a valid legislative session. (2017-2018 for all states, 2015-2016 for CA)
  if years is not None and ((not all(item > 2015 for item in years) and "CA" not in states) or (not all(item <= 2018 for item in years))):
    raise Exception("""Data for requested year not included in corpus.
     Valid session_years are 2017 and 2018 for all states provided. 2015 and 2016 are valid years for CA.""")

  payload = {}
  header_row = True

  #If no states specified, retrieve relevant files for all valid states
  if states is None:
    states = VALID_STATES

  #If no years/session specified, retrieve data for all valid state legislative session years
  if years is None:
    if "CA" in states:
      years= [2015,2016,2017,2018]
    else:
      years = [2017,2018]

  #The following code block operates as follows:
  # For every state and year requested, read the relevant CSV file(s), then
  # load it into a python object (payload) which is returned to user
  for state in states:
    FILE_PATHS = []

    #Build the filepaths to the correct data location given the states and years provided
    #Years 2017 and 2018 are valid inputs that belong to the same 2017-2018 session
    if 2017 in years or 2018 in years:
      FILE_PATHS.append(CORPUS_FILE_PATH + state + "/2017-2018/CSV/" + file_name + ".csv")

    #CA has 2 valid legislative sessions (2015-2016 and 2017-2018)
    #This means the entirety of CA data is located in more than one folder, unlike other states.
    #Looping through a list of filepaths allows us to handle this corner case
    if state == "CA" and (2015 in years or 2016 in years):
      FILE_PATHS.append(CORPUS_FILE_PATH + state + "/2015-2016/CSV/" + file_name + ".csv")

    for FILE_PATH in FILE_PATHS:
      #Open the file to read
      with open(FILE_PATH, newline='', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        #Read CSV row by row
        for row in rows:
          #The first row of every CSV we visit is the header row, containing the names for each column
          # We will add this to the payload only once, as every CSV we read after this will be the same headers
          if header_row:
            payload['column_headers'] = row
            #Sets up 'rows' in payload where we will store future records
            payload['rows'] = []
            header_row = False
            continue
          #Load CSV Into payload row by row
          payload['rows'].append(row)

  return payload

#Retrieve Committee Name & ID, hearing date, list of videos of the hearing, and state for a given HID
#Input:
#  Required: HID (type:Positive Int) (Ex: 10003)
#  Optional: speeches
#      -If searching through from a specific state or session, pass in
#         speeches=load_content("speeches", specific states, specific years)
#Output:
#   Dictionary
#   If no matches exist does not exist in database, returns []
def get_metadata_hearing(hid, hearings=None, videos=None):
  HID_IDX=0
  CID_IDX=4
  CNAME_IDX=8
  HDATE_IDX=1
  STATE_IDX=3

  if hearings is None:
    hearings=load_content("hearings")
  hid = str(hid)

  hearingData=None
  for row in hearings['rows']:
    if hid == row[HID_IDX]:
      hearingData={'hid':row[HID_IDX],'cid':row[CID_IDX],'cname':row[CNAME_IDX],'hearing_date':row[HDATE_IDX],'state':row[STATE_IDX]}
  if hearingData is None:
    return {}

  #vids, videos = videos_from_hid(hid)
  #hearingData['vids'] = vids
  hearingData['videos'] = videos_from_hid(hid)

  return hearingData


#Retrieve the video ids and video URLs associated with a hearing
#Input:
#  Required: Hearing ID (type:Positive Int) (Ex: 10003)
#  Optional: videos
#      -If searching through from a specific state or session, pass in
#         videos=load_content("videos", specific states, specific years)
#Output:
#   (Type: List of[[Int,String]]) (Ex: [[0, "video0.mp4"], [1,"video1.mp4"]])
#   If no matches exist does not exist in database, returns []
def videos_from_hid(hid, videos=None):
  HID_IDX = 2
  VID_IDX = 0
  URL_IDX = 7

  if videos is None:
    videos=load_content("videos")

  vids = []
  video_files=[]
  hid=str(hid)

  for row in videos['rows']:
    if row[HID_IDX] == hid:
      vids.append((row[VID_IDX],row[URL_IDX])) #tuple of (video id, url)
  return vids

#Create transcript for a given bill discussion with metadata
#Input:
#  Required: HID (type:Positive Int) (Ex: 10003)
#  Required: BID (type: String)
#  Optional: speeches
#      -If searching through from a specific state or session, pass in
#         speeches=load_content("speeches", specific states, specific years)
#Output:
#   Dictionary and Transcript
#   If no matches exist does not exist in database, returns []
def get_hearing_transcript(hid, bid, speeches=None):
  PID_IDX=1
  BID_IDX=4
  HID_IDX=3
  VID_START_IDX = 9
  VID_END_IDX = 10
  LAST_NAME_IDX = 14
  FIRST_NAME_IDX = 15
  TEXT_IDX = 16
  STARTING_TIME_IDX = 11
  if speeches is None:
    speeches=load_content("speeches")

  hid=str(hid)

  lines = []
  for row in speeches['rows']:
    if hid == row[HID_IDX] and bid == row[BID_IDX]:
      offset_time = add_seconds("00:00:00", int(row[STARTING_TIME_IDX]))
      line = {'video start':row[VID_START_IDX],'video end':row[VID_END_IDX],'offset':offset_time,'bid':row[BID_IDX],
              'first name':row[FIRST_NAME_IDX],'last name':row[LAST_NAME_IDX],'pid':row[PID_IDX],'text':row[TEXT_IDX]}
      lines.append(line)
  return lines

#Helper function to get bill discussion metadata
def bill_discussion_info(hid, bid, hearings=None, speeches=None, videos=None):
  return {"metadata":get_metadata_hearing(hid, hearings,videos), "transcript":get_hearing_transcript(hid, bid, speeches)}


# accepts the output from get_hearing_transcript() and prints a transceript.
def pprint_discussion(metadata, transcript_info):
  videos = {}
  for vid, url in metadata['videos']:
    videos[vid] = url

  print()
  print(f"] Discussion of {metadata['state']} {metadata['cname']} held on {metadata['hearing_date']}")
  print(f"] {len(videos.keys())} videos")
  print("] printing transcript: ")

  prev_video = -1
  for line in transcript_info:
    video = line['video start']
    if video != prev_video:
      print()
      print(f"] Discussing {line['bid']}")
      print(f"] Video: {videos[line['video start']] if line['video start'] in videos else line['video start']}")
      print()
      prev_video = video
    print(f"[{line['offset']}] {line['first name']} {line['last name']}: ")
    print(f"\t{line['text']}")
  print()

discussion = bill_discussion_info(254743, "CA_201720180AB2411")

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


def create_speakers():
  speakers = {}

  for speaker in discussion['transcript']:
    name = speaker['first name'] + " " + speaker['last name']

    if name not in speakers:
      speakers[name] = {'utterances': [], 'sentiment': []}
    speakers[name]['utterances'].append(speaker['text'])
    speakers[name]['sentiment'].append(get_sentiment(speaker['text']))

  for speaker in speakers:
    speakers[speaker]['avg_sentiment'] = get_avg_sentiment(speakers[speaker]['sentiment'])

  return speakers


def get_utterances_summary_large_book(speaker):
  utterances = speakers[speaker]['utterances']

  summarizer = Summarizer(model_name_or_path="pszemraj/led-large-book-summary", token_batch_length=4096,)
  return summarizer.summarize_string(speaker + ':' + ' '.join(utterances))


def get_utterances_summary_bart(speaker):
  utterances = speakers[speaker]['utterances']

  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  summary = summarizer(' '.join(utterances), do_sample=False)[0]['summary_text']
  return summary


def find_min_max_speaker_sentiment():
  min_speaker, max_speaker = None, None
  min_avg_sentiment, max_avg_sentiment = float('inf'), float('-inf')

  for speaker, speaker_data in speakers.items():
    avg_sentiment = speaker_data['avg_sentiment']
    
    if avg_sentiment < min_avg_sentiment:
      min_avg_sentiment = avg_sentiment
      min_speaker = speaker
    
    if avg_sentiment > max_avg_sentiment:
      max_avg_sentiment = avg_sentiment
      max_speaker = speaker

  return (min_speaker, min_avg_sentiment), (max_speaker, max_avg_sentiment)


def find_min_max_speaker_utterances():
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

speakers = create_speakers()

def print_stats():
  print(f'Num speakers: {len(speakers)}')
  total_num_utterances = sum([len(v['utterances']) for v in speakers.values()])
  print(f'Num utterances: {total_num_utterances}')
  print('\n')

  for speaker in speakers:
    avg_sentiment = speakers[speaker]['avg_sentiment']
    num_utterances = len(speakers[speaker]['utterances'])

    print(f'{speaker} spoke {num_utterances} / {total_num_utterances} times')
    print(f'Avg sentiment of speaker {speaker}: {avg_sentiment}')
    print('\n')

print_stats()

(min_speaker, min_avg_sentiment), (max_speaker, max_avg_sentiment) = find_min_max_speaker_sentiment()
print(f"{min_speaker} was the most negative speaker during the hearing.")
print(f"{max_speaker} was the most positive speaker during the hearing.")

(min_utterances_speakers, min_utterances), (max_utterances_speakers, max_utterances) = find_min_max_speaker_utterances()
print(f"{min_utterances_speakers} spoke the least during the hearing, speaking only {min_utterances} times.")
print(f"{max_utterances_speakers} spoke the most during the hearing, speaking {max_utterances} times.")

import nltk
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import re
import logging

logging.getLogger('gensim').setLevel(logging.ERROR)

nltk.download('stopwords', quiet=True)

texts = [entry["text"] for entry in discussion['transcript']]

from huggingface_hub import InferenceClient
client = InferenceClient(api_key="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")

def get_utterances_summary_mistral(speaker):
  utterances = speakers[speaker]['utterances']

  messages = [
    {
      "role": "user",
      "content": f"Provide a concise two sentence analysis of {speaker}'s opinion given their utterances: {' '.join(utterances)}"
    }
  ]

  completion = client.chat.completions.create(model="mistralai/Mistral-Nemo-Instruct-2407", messages=messages, max_tokens=500)
  category = completion.choices[0].message.content

  return category

print(get_utterances_summary_mistral(min_speaker))
print(get_utterances_summary_mistral(max_speaker))

def preprocess(text, stop_words):
  result = []
  for token in simple_preprocess(text, deacc=True):
    if token not in stop_words and len(token) > 3:
      result.append(token)
  return result

def get_topic_lists_from_pdf():
  names = set()
  for speaker in discussion['transcript']:
    names.add(speaker['first name'].lower())
    names.add(speaker['last name'].lower())

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

topic_list = get_topic_lists_from_pdf()

categories = set()
for topic in topic_list:
  categories.add(get_political_category_from_topic_lists(topic))

print(f'Topics discussed during the hearing included {categories}.')