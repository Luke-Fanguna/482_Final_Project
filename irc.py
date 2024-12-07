import os
import socket
import sys
import time
import re
import time
import select
import mysql.connector
from dotenv import load_dotenv
import random as rand
import csv

# TODO: update this file to reflect google doc requirements
# TODO: return orgs also not mentioned in DB-- filter these with ORG, or INC, or institution etc. don't want random strings included
# list [number] shows all possible did for a given number. ex: [list 141] -> 14111, 141453, etc. basically all possible did given a number
# show [did] will show whether a phenom was detected in a hearing


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
        self.channel = "##ScottPramukChannelTest"
        self.botnick = f"sdl-bot"

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
        # Need to get a list of all dids
        # TODO what to do when the list is too long to display all values? Pick a random sample?
        res = []
        for key in self.didMap.keys():
            if str(key).startswith(str(num)):
                res.append(key)
        if not res:
            self.irc.send(self.channel, f"{
                userName}: No entries in Utterance table with DID that starts with {num}. Instead, try {self.valid_did_stem(num)}")
            return
        if len(res) > 40:
            res = rand.sample(res, 40)
            self.irc.send(self.channel, f"{
                userName}: 40 Randomly selected possible DID values:")
        else:
            self.irc.send(self.channel, f"{
                userName}: Possible DID values:")
        for i in range(0, len(res), 10):
            if i + 10 < len(res):
                self.irc.send(self.channel, f"{
                    userName}: {res[i:i+10]}")
            else:
                self.irc.send(self.channel, f"{
                    userName}: {res[i:]}")

    def on_show(self, did, userName):
        if not did.isdigit():
            self.irc.send(self.channel, f"{
                userName}: DID must be an integer.")
            return
        elif did.isdigit() and int(did) not in self.didMap.keys():
            self.irc.send(self.channel, f"{
                userName}: Invalid did selected.")
            return
        # Collect all Utterance with the specified did
        utterances = self.collectUtterance(did)
        if not utterances:
            self.irc.send(self.channel, f"{
                userName}: No utterances with this did were found.")
        # TODO do NLP tasks on this list of utterances
        self.irc.send(self.channel, f"{
            userName}: Utterances found. Bill discussed: {self.didMap[int(did)][0]} ")
        print(utterances)

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

    def generateBIDs(self):
        # Idea: create dict with all unique dids as keys, (bid,hid) as values
        # Open the CSV file for reading
        with open('map.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the header
            next(reader)
            # Iterate over each row
            i = 0
            for row in reader:
                i += 1
                did, bid, hid = row
                if i < 10:
                    print(f'did: {did}, bid: {bid}, hid: {hid}')

                self.didMap[int(did)] = (str(bid), int(hid))

    def on_quit(self, userName):
        self.irc.send(self.channel, f"{
                      userName}: You are mean. I'm leaving now then.")
        self.irc.command("QUIT")
        sys.exit()

    def on_who(self, userName):
        resp1 = f"{userName}: My name is {
            self.botnick}. I was created by Scott, Luke, Daniel, CSC 482-01"
        # resp2 = f"{userName}: I can generate a random picture of a cat for you with the command: [{self.botnick}: iluvcats]"
        resp2 = f"{userName}: Use the command: [{
            self.botnick}: list [integer]] to get a list of valid discussion IDs that begin with the digits in the integer."
        resp3 = f"{userName}: Use the command: [{
            self.botnick}: show [did]] where did is a valid discussion ID and I will output all phenoms detected in utterances tagged with this did."
        self.irc.send(self.channel, resp1)
        self.irc.send(self.channel, resp2)
        self.irc.send(self.channel, resp3)

    def get_username_from_text(self, text):
        return text[text.index(':') + 1: text.index("!")].replace("@", "")

    def run(self):
        while True:
            text = self.irc.get_response()
            print("RECEIVED ==> ", text)
            # should the usage of a command reset the start time? would allow for more cmds
            if text and "PRIVMSG" in text and self.channel in text and self.botnick + ":" in text:
                userName = self.get_username_from_text(text)
                if re.search(rf'{self.botnick}:\s*(die)', text, re.IGNORECASE) and '-bot' not in userName:
                    self.on_quit(userName)
                elif re.search(rf'{self.botnick}:\s*(forget)', text, re.IGNORECASE):
                    self.on_forget(userName)
                elif re.search(rf'{self.botnick}:\s*(who are you|usage)', text, re.IGNORECASE):
                    self.on_who(userName)

                elif re.search(rf'{self.botnick}:\s*(list)\s*(.*)', text, re.IGNORECASE):
                    print(f"List command:")
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
