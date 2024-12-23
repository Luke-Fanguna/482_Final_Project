import os
import socket
import sys
import time
import re
import time
import select
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import random as rand
import csv
from sentiment_analysis_irc import get_phenoms
from toxic import toxic_phenom
from org_detection import bertModel
from transformers import AutoTokenizer, AutoModelForTokenClassification


class IRC:
    def __init__(self):
        # Define the socket
        self.irc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def command(self, msg):
        self.irc.send(bytes(msg + "\n", "UTF-8"))

    def send(self, channel, msg):
        # Transfer data
        self.command("PRIVMSG " + channel + " :" + msg)

    def connect(self, server, port, channel, botnick):
        # Connect to the server
        print("Connecting to: " + server)
        self.irc.connect((server, port))

        # Perform user authentication
        self.command("USER " + botnick + " " +
                     botnick + " " + botnick + " :python")
        self.command("NICK " + botnick)
        time.sleep(5)

        # join the channel
        self.command("JOIN " + channel)

    def get_response(self, timeout=20):
        ready_sockets, x, y, = select.select([self.irc], [], [], timeout)
        if ready_sockets:
            # Data is available to read
            resp = self.irc.recv(2040).decode("UTF-8")

            if 'PING' in resp:
                self.command('PONG ' + resp.split()[1] + '\r')

            return resp
        else:
            # Timeout occurred
            return None


class Bot():
    def __init__(self):
        # IRC Config

        load_dotenv()
        server = "irc.libera.chat" 	# Provide a valid server IP/Hostname
        port = 6667
        self.channel = "##luke-testing-bot"
        self.botnick = f"sdl-bot"
        self.orgModel = bertModel(3, model=AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER"), tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER"))
        self.numConversations = 0
        self.choices = set()
        self.cursor = None
        self.connection = None
        self.didMap = dict()

        self.createConnection()
        self.generateBIDs()

        self.irc = IRC()
        self.irc.connect(server, port, self.channel, self.botnick)

    def createConnection(self):
        self.connection = mysql.connector.connect(
            host=os.getenv("myhost"),
            user=os.getenv("myuser"),
            password=os.getenv("mypass"),
            database=os.getenv("mydb"),
            connection_timeout=300,  # Set the connection timeout in seconds
            buffered=True            # Ensures complete result sets are fetched
        )

    def valid_did_stem(self, num):
        # Function to find a stem of did that is close to the number entered by user
        maxDist = 100000000
        closeDID = None
        for key in self.didMap.keys():
            if abs(int(str(key)[0:len(str(num))]) - int(num)) < maxDist:
                closeDID = str(key)[0:len(str(num))]
                maxDist = abs(int(str(key)[0:len(str(num))]) - int(num))
        return closeDID

    def on_list(self, num, userName):

        res = []
        for key in self.didMap.keys():
            if str(key).startswith(str(num)):
                res.append(key)
        if not res:
            self.irc.send(self.channel, f"{userName}: No entries in Utterance table with DID that starts with {num}. Instead, try {self.valid_did_stem(num)}")
            return
        if len(res) > 40:
            res = rand.sample(res, 40)
            self.irc.send(self.channel, f"{userName}: 40 Randomly selected possible DID values:")
        else:
            self.irc.send(self.channel, f"{userName}: Possible DID values:")
        for i in range(0, len(res), 10):
            if i + 10 < len(res):
                self.irc.send(self.channel, f"{userName}: {res[i:i+10]}")
            else:
                self.irc.send(self.channel, f"{userName}: {res[i:]}")

    def on_show(self, did, userName):
        if not did.isdigit():
            self.irc.send(self.channel, f"{userName}: DID must be an integer.")
            return
        elif did.isdigit() and int(did) not in self.didMap.keys():
            self.irc.send(self.channel, f"{userName}: Invalid did selected.")
            return
        # Collect all Utterance with the specified did
        utterances = self.collectUtterance(did)
        if not utterances:
            self.irc.send(self.channel, f"{userName}: No utterances with this did were found.")
            return

        bill = self.didMap[int(did)][0]

        self.irc.send(self.channel, f"{userName}: Utterances found. Bill discussed: {bill} ")
        self.irc.send(self.channel, f"{userName}: Summary of {bill}: {self.get_summary(bill)}")

        people = self.collectPeopleFromUtterances(utterances)
        x = self.orgModel.processUtterance(
            utterances, self.peopleFromUtterancesNoneAllowed(utterances)).items()
        peopleCounter = 0
        for (last_name, first_name), orgs in x:
            peopleCounter += 1
            curString = ""
            for org, count in orgs.items():
                if curString == "":
                    curString = f"{userName}: {first_name} {last_name} mentioned '{org}' {count} time{'s' if count > 1 else ''}"
                else:
                    curString += f", '{org}' {count} time{'s' if count > 1 else ''}"
            self.irc.send(self.channel, curString)
            if peopleCounter % 3 == 0:
                time.sleep(3)
        if people:
            sentiment_phenoms = get_phenoms(utterances, people)
            for phenom in sentiment_phenoms:
                self.irc.send(self.channel, f"{userName}: {phenom} ")
        else:
            self.irc.send(self.channel, f"{userName}: Some utterances are unlabeled. Skipping detection of some phenoms.")

        toxic = toxic_phenom(did)
        if toxic:
            for t in toxic:
                self.irc.send(self.channel, f"{userName}: {t}")
        else:
            self.irc.send(self.channel, f"{userName}: Toxic Phenom was not detected")

    def peopleFromUtterancesNoneAllowed(self, utterances):
        pids = set([pid for _, pid in utterances])
        if len(pids) == 0:
            return []

        person_ids = ', '.join(['%s'] * len(pids))

        self.cursor = self.connection.cursor()
        self.cursor.execute(
            f"SELECT pid,last,first FROM Person WHERE pid IN ({person_ids})", list(pids))
        result = self.cursor.fetchall()
        self.cursor.close()

        people_map = {pid: (last.strip(), first.strip())
                      for pid, last, first in result}
        return people_map

    def collectPeopleFromUtterances(self, utterances):
        pids = set([pid for _, pid in utterances])

        if None in pids:
            return None
        if len(pids) == 0:
            return []

        person_ids = ', '.join(['%s'] * len(pids))

        self.cursor = self.connection.cursor()
        self.cursor.execute(
            f"SELECT pid,last,first FROM Person WHERE pid IN ({person_ids})", list(pids))
        result = self.cursor.fetchall()
        self.cursor.close()

        people_map = {pid: (last.strip(), first.strip())
                      for pid, last, first in result}
        return people_map

    def collectUtterance(self, did):
        utterances = []
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            f"SELECT text,pid FROM Utterance where did = {did}")
        rows_table = self.cursor.fetchall()  # Fetch the results of the first query
        self.cursor.close()
        for row in rows_table:
            # list of the form: (text,pid of speaker)
            utterances.append((row[0], row[1]))
        return utterances
    def get_summary(self, bid):
        print(type(bid))
        sql = f"""
            SELECT summary_text FROM BillAnalysis
            WHERE bid = '{bid}';
                """
        summary = ""
        try:
            # Open a cursor
            self.cursor = self.connection.cursor()
            self.cursor.execute(sql)
            summary = self.cursor.fetchall()
            if summary:
                summary = summary[0][0]
            else:
                summary = f"No summary provided for {bid}"
            print(summary)
            print("good")
        except Error as e:
            print("bad", e)
            summary = f"No summary provided for {bid}"
        finally:
            if self.cursor:
                self.cursor.close()
        return summary

    def generateBIDs(self):
        # Idea: create dict with all unique dids as keys, (bid,hid) as values
        # Open the CSV file
        with open('map.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Skip header
            next(reader)
            # Iterate over each row in CSV
            for row in reader:
                did, bid, hid = row
                self.didMap[int(did)] = (str(bid), int(hid))

    def on_quit(self, userName):
        self.irc.send(self.channel, f"{userName}: You are mean. I'm leaving now then.")
        self.irc.command("QUIT")
        sys.exit()

    def on_who(self, userName):
        resp1 = f"{userName}: My name is {self.botnick}. I was created by Scott, Luke, Daniel, CSC 482-01"
        resp2 = f"{userName}: Use the command: [{self.botnick}: list [integer]] to get a list of valid discussion IDs that begin with the digits in the integer."
        resp3 = f"{userName}: Use the command: [{self.botnick}: show [did]] where did is a valid discussion ID and I will output all phenoms detected in utterances tagged with this did."
        self.irc.send(self.channel, resp1)
        self.irc.send(self.channel, resp2)
        self.irc.send(self.channel, resp3)

    def get_username_from_text(self, text):
        return text[text.index(':') + 1: text.index("!")].replace("@", "")

    def run(self):
        while True:
            text = self.irc.get_response()
            print("RECEIVED ==> ", text)

            if text and "PRIVMSG" in text and self.channel in text and self.botnick + ":" in text:
                userName = self.get_username_from_text(text)
                if re.search(rf'{self.botnick}:\s*(die)', text, re.IGNORECASE) and '-bot' not in userName:
                    self.on_quit(userName)
                elif re.search(rf'{self.botnick}:\s*(forget)', text, re.IGNORECASE):
                    self.on_forget(userName)
                elif re.search(rf'{self.botnick}:\s*(who are you|usage)', text, re.IGNORECASE):
                    self.on_who(userName)

                elif re.search(rf'{self.botnick}:\s*(list)\s*(.*)', text, re.IGNORECASE):
                    # Regex to extract everything after 'list'
                    pattern = r"(?<=list\s).*"
                    # Perform regex search
                    match = re.search(pattern, text)
                    if match:
                        result = match.group(0).strip()
                        self.on_list(result, userName)
                elif re.search(rf'{self.botnick}:\s*(show)\s*(.*)', text, re.IGNORECASE):
                    # Regex to extract everything after 'list'
                    pattern = r"(?<=show\s).*"
                    # Perform regex search
                    match = re.search(pattern, text)
                    if match:
                        result = match.group(0).strip()
                        self.on_show(result, userName)
                else:
                    continue


if __name__ == "__main__":
    bot = Bot()
    bot.run()
