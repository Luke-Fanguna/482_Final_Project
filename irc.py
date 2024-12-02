import os
import socket
import sys
import time
import re
import time
import select
import mysql.connector
from dotenv import load_dotenv


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

        self.irc = IRC()
        self.irc.connect(server, port, self.channel, self.botnick)

        self.numConversations = 0
        self.choices = set()
        self.cursor = None
        self.connection = None

        self.createConnection()
        self.generateChoices()

    def createConnection(self):
        self.connection = mysql.connector.connect(
            host=os.getenv("SQL_DB_host"),
            user=os.getenv("SQL_DB_user"),
            password=os.getenv("SQL_DB_password"),
            database=os.getenv("SQL_DB_database"),
            connection_timeout=300,  # Set the connection timeout in seconds
            buffered=True            # Ensures complete result sets are fetched
        )

    def generateChoices(self):
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT bid,hid FROM BillDiscussion limit 20")
        rows_table = self.cursor.fetchall()  # Fetch the results of the first query
        self.cursor.close()
        for row in rows_table:
            self.choices.add((row[0], row[1]))

    def on_quit(self, userName):
        self.irc.send(self.channel, f"{
                      userName}: You are mean. I'm leaving now then.")
        self.irc.command("QUIT")
        sys.exit()

    def on_who(self, userName):
        if not self.choices:
            self.cursor = self.connection.cursor()
            self.cursor.execute("SELECT bid,hid FROM BillDiscussion limit 20")
            rows_table = self.cursor.fetchall()  # Fetch the results of the first query
            self.cursor.close()
            for row in rows_table:
                self.choices.add((row[0], row[1]))
        resp1 = f"{userName}: My name is {
            self.botnick}. I was created by Scott, Luke, Daniel, CSC 482-01"
        # resp2 = f"{userName}: I can generate a random picture of a cat for you with the command: [{self.botnick}: iluvcats]"
        resp3 = f"{userName}: Here is a list of Bill ID, Hearing ID tuples: {
            list(self.choices)[5]}. Choose a valid entry and I will find the phenoms from the hearing transcript! Use the command: [{self.botnick}: phenoms: BID, HID]"
        self.irc.send(self.channel, resp1)
        # self.irc.send(self.channel, resp2)
        self.irc.send(self.channel, resp3)

    def on_phenom(self, bid, hid):
        self.cursor = self.connection.cursor()
        self.cursor.execute(f"SELECT did FROM BillDiscussion where bid = '{
                            bid}' and hid = {hid}")
        rows_table = self.cursor.fetchall()  # Fetch the results of the first query
        self.cursor.close()
        curDid = None
        for row in rows_table:
            curDid = row[0]
        self.cursor = self.connection.cursor()
        self.cursor.execute(f"SELECT text FROM Speeches where did = {
                            curDid}")
        rows_table = self.cursor.fetchall()  # Fetch the results of the first query
        allText = []
        for row in rows_table:
            print(row[0])
            allText.append(row[0])
        # allText is a list with each element being the text of a speech
        # We can run our phenom detection on each element which will allow us to also return the element that a certain phenom is linked to
        self.cursor.close()

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
                elif re.search(rf'{self.botnick}:\s*(phenom):\s*(.*)', text, re.IGNORECASE):
                    # Regex to extract the content after 'phenom:'
                    pattern = r"(?<=phenom: )([\w\s,]+)"
                    # Perform regex search
                    match = re.search(pattern, text)
                    if match:
                        result = [item.strip()
                                  for item in match.group(1).split(",")]
                        print("Extracted values:", result)
                        if (result[0], int(result[1])) in self.choices:
                            self.irc.send(
                                self.channel, "Valid choice selected. Generating phenoms!")
                            self.on_phenom(result[0], result[1])

                    else:
                        print("No match found")
                else:
                    continue


if __name__ == "__main__":
    bot = Bot()
    bot.run()
