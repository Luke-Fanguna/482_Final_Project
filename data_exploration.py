# IDEA to find the average length of all utterances and ratio of utterances with orgs vs those without
from transformers import AutoTokenizer, AutoModelForTokenClassification
from Levenshtein import distance as levenshtein_distance
from transformers import pipeline
import csv
import os
import re
import pandas as pd
import random as rand

import mysql.connector
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForTokenClassification


class bertModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def createTaggedOutputs(self, utterance):
        tagged_outputs = []
        ner_results = self.nlp(utterance)
        tagged_outputs.append(ner_results)
        return tagged_outputs

    def isOrgDetected(self, utterance):
        ner_results = self.nlp(utterance)
        if len(ner_results) < 1:
            return False
        for entry in ner_results:
            if entry["entity"] == 'B-ORG' or entry["entity"] == 'I-ORG':
                return True
        return False

    def clean_word(self, word):
        """Clean up unwanted ## patterns from the word."""
        word = re.sub(r' ##', '', word)  # Remove ' ##' sequences
        word = re.sub(r'^##', '', word)  # Remove '##' at the start
        return word

    def merge_org_entities(self, ner_output):
        merged_entities = []
        current_org = None
        for el in ner_output:
            for entry in el:
                if entry['entity'] == 'B-ORG':
                    # Start a new organization
                    if current_org:
                        current_org['word'] = self.clean_word(
                            current_org['word'])
                        merged_entities.append(current_org)
                    current_org = {
                        'entity': 'ORG',
                        'score': entry['score'],
                        'word': entry['word'],
                        'start': entry['start'],
                        'end': entry['end']
                    }
                elif entry['entity'] == 'I-ORG' and current_org:
                    # Append to the current organization
                    current_org['word'] += f" {entry['word']}"
                    # Take the minimum score
                    current_org['score'] = min(
                        current_org['score'], entry['score'])
                    current_org['end'] = entry['end']
                else:
                    # Finalize the current organization if it exists
                    if current_org:
                        current_org['word'] = self.clean_word(
                            current_org['word'])
                        merged_entities.append(current_org)
                        current_org = None
                    # Add non-ORG entities directly
                    merged_entities.append(entry)

            # Add the last organization if it's not already added
            if current_org:
                current_org['word'] = self.clean_word(current_org['word'])
                merged_entities.append(current_org)
                current_org = None
        return merged_entities

    def validate_org_entities(self, merged_ents, csv_file_path):
        validated_orgs = dict()

        # Read the CSV file values
        with open(csv_file_path, 'r') as file:
            csv_values = [row[1]
                          # Assuming single-column CSV
                          for row in csv.reader(file)]

        # Iterate over model outputs
        for entry in merged_ents:
            entity_type = entry['entity']
            text = entry['word']
            if text in validated_orgs:
                # Increase # of times org was mentioned
                validated_orgs[text] = validated_orgs[text] + 1
                continue

            # Check if ORG, concatenated B- and I- ORG above
            # Just checking if org seen here for rough estimate on data
            if entity_type == "ORG":
                validated_orgs[text] = 1

        return validated_orgs


class Connection:
    def __init__(self):
        load_dotenv()
        self.connection = mysql.connector.connect(
            host=os.getenv("myhost"),
            user=os.getenv("myuser"),
            password=os.getenv("mypass"),
            database=os.getenv("mydb"),
            connection_timeout=300,  # Set the connection timeout in seconds
            buffered=True            # Ensures complete result sets are fetched
        )


class sampleData:
    def __init__(self, connection, model: bertModel, chunk_size=10000, sample_size=100, percent_with_org=100, desired_word_length=88.35432):
        # Note above values taken from metrics gathered from 100,000 randomly sampled utterances
        self.connection = connection
        self.chunk_size = chunk_size
        self.sample_size = sample_size
        self.percent_with_org = percent_with_org
        self.data = pd.DataFrame(
            columns=['utterance', 'word_count', 'org_detected'])
        self.model = model
        self.desired_word_length = desired_word_length

    def detect_organization(self, text):
        # Check if org detected in this text
        return self.model.isOrgDetected(text)

    def fetch_data(self):
        cursor = self.connection.connection.cursor()
        cursor.execute(f"SELECT text FROM Utterance ORDER BY RAND() LIMIT {
                       self.chunk_size}")
        rows_chunk = cursor.fetchall()
        cursor.close()
        # row[0] contains the text i.e. utterance
        return [row[0] for row in rows_chunk]

    def process_data(self, utterances):
        # Putting new entries into a DF
        chunk_data = pd.DataFrame({'utterance': utterances})
        chunk_data['word_count'] = chunk_data['utterance'].apply(
            lambda x: len(x.split()))
        chunk_data['org_detected'] = chunk_data['utterance'].apply(
            self.detect_organization)
        self.data = pd.concat([self.data, chunk_data], ignore_index=True)

    def stratified_sample(self):
        # Stratified sample
        # Prioritze words with length close to suggested average length
        samples_with_org = int(self.percent_with_org * self.sample_size)
        samples_without_org = self.sample_size - samples_with_org

        org_detected_group = self.data[self.data['org_detected'] == 1]
        org_not_detected_group = self.data[self.data['org_detected'] == 0]

        # Column for differences between word length and desired length
        org_detected_group['word_diff'] = abs(
            org_detected_group['word_count'] - self.desired_word_length)
        org_not_detected_group['word_diff'] = abs(
            org_not_detected_group['word_count'] - self.desired_word_length)

        # Sort by word difference and then sample
        org_detected_group = org_detected_group.sort_values(by='word_diff')
        org_not_detected_group = org_not_detected_group.sort_values(
            by='word_diff')

        sample_with_org = org_detected_group.head(
            min(samples_with_org, len(org_detected_group)))
        sample_without_org = org_not_detected_group.head(
            min(samples_without_org, len(org_not_detected_group)))

        final_sample = pd.concat([sample_with_org, sample_without_org])
        return final_sample

    def save_sample(self, final_sample, file_name='sample_utterances_containing_orgs.csv'):
        print(f"saving to csv...: {file_name}")
        final_sample.to_csv(file_name, index=False)


def main():
    import os
    print("Current working directory:", os.getcwd())
    # Create connection to SQL DB
    connection = Connection()
    # Init bert model
    bert = bertModel()
    sample = sampleData(connection, bert, 10000, 200)
    utterances = sample.fetch_data()
    sample.process_data(utterances)
    final_sample = sample.stratified_sample()
    sample.save_sample(final_sample)

    # Verify sample statistics
    sample_avg_words = final_sample['word_count'].mean()
    sample_percent_with_org = final_sample['org_detected'].mean()

    print("Sample Stats:")
    print(f"Average words per utterance: {sample_avg_words}")
    print(f"Percent with org detected: {sample_percent_with_org}")


# Call the main function
if __name__ == "__main__":
    main()
