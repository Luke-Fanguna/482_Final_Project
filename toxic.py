import pandas as pd
import sys
import mysql.connector
import os
from dotenv import load_dotenv
import datetime
from detoxify import Detoxify
load_dotenv()


def get_discussion(did):
    table = pd.read_csv('utterance2324.csv')
    discussion = table[table['did'] == float(did)]
    return discussion

def find_phenom(table):
    uid = []
    for i, t in enumerate(table['text']):
        results = Detoxify('original').predict(t)
        high = max(results.values()) > .1
        if high:
            uid.append(table.iloc[i]['uid'])
    return uid

def get_metadata(did):
    sql = """
    SELECT 
        bid, lastTouched
    FROM
        BillDiscussion
    WHERE did = %s;
    """

    try:
        db = mysql.connector.connect(
            host=os.getenv("myhost"),
            user=os.getenv("myuser"),
            password=os.getenv("mypass"),
            database=os.getenv("mydb")
        )
        cur = db.cursor()
        cur.execute(sql, (did,))

        bill, date = cur.fetchall()[0]
        date = date.strftime('%m/%d/%Y')
        print("Bill discussed:", bill)
        print("Date of discussion:", date)

        db.close()

    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None, None

def get_person(pid):
    sql = """
    SELECT 
        first, last
    FROM Person
    WHERE pid = %s;
    """

    try:
        db = mysql.connector.connect(
            host=os.getenv("myhost"),
            user=os.getenv("myuser"),
            password=os.getenv("mypass"),
            database=os.getenv("mydb")
        )
        cur = db.cursor()
        cur.execute(sql, (pid,))

        first, last = cur.fetchall()[0]
        db.close()

    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None, None

    return first + " " + last


def get_label(text):
    scores = pd.read_csv('scores.csv')
    label = scores[scores['text'] == text]['reason'].iloc[0]
    return label


def create_phenom(uids):
    # TODO: Format as { PERSON said THIS, flagged as LABEL }
    utterance = pd.read_csv('utterance2324.csv')
    output = []
    for uid in uids:
        person = get_person(utterance[utterance['uid'] == uid]['pid'].iloc[0])
        text = utterance[utterance['uid'] == uid]['text'].iloc[0]
        label = get_label(text)

        output.append(f'{person} was tagged for {label} by saying {text}')
    return output

def toxic_phenom(did):
    get_metadata(did)
    table = get_discussion(did)
    uids = find_phenom(table)
    if uids:
        return create_phenom(uids)
    else:
        return None
