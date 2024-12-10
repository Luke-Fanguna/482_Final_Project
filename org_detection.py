
from transformers import AutoTokenizer, AutoModelForTokenClassification
from Levenshtein import distance as levenshtein_distance
from transformers import pipeline
import csv
import os
import re
from ast import literal_eval
import mysql.connector
from dotenv import load_dotenv
from statistics import mean
from transformers import AutoTokenizer, AutoModelForTokenClassification
import nltk

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree


class bertModel:
    def __init__(self, editDist, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        self.editDist = editDist

    def createTaggedOutputs(self, utterance):
        tagged_outputs = []
        ner_results = self.nlp(utterance)
        tagged_outputs.append(ner_results)
        return tagged_outputs

    def clean_word(self, word):
        """Clean up unwanted ## patterns from the word."""
        word = re.sub(r' ##', '', word)  # Remove ' ##' sequences
        word = re.sub(r'^##', '', word)  # Remove '##' at the start
        return word

    def merge_org_entities_trained_model(self, ner_output):
        merged_entities = []
        current_org = None
        for el in ner_output:
            for entry in el:
                # ORG label renamed to LABEL_5
                if entry['entity'] == 'LABEL_5' and not current_org:
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
                elif entry['entity'] == 'LABEL_5' and current_org and current_org['end'] == entry['start']:
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
        org_keywords = ['LLC', 'Inc.', 'Corp.', 'Limited', 'Ltd.', 'Co.', 'Foundation',
                        'Association', 'Institute', 'University', 'College', 'Group', 'Enterprise', 'Holding',
                        'Ventures', 'Partners']
        # Create a regex pattern to match any of the keywords, case-insensitive
        pattern = r'\b(?:' + '|'.join(re.escape(keyword)
                                      for keyword in org_keywords) + r')\b'

        # Read the CSV file values
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_values = [row[1]
                          for row in csv.reader(file)]

        # Iterate over outputs
        for entry in merged_ents:
            entity_type = entry['entity']
            text = entry['word']
            if text in validated_orgs:
                # Increase # of times org was mentioned
                validated_orgs[text] = validated_orgs[text] + 1
                continue

            # Check if ORG, concatenated B- and I- ORG above
            if entity_type == "ORG":
                # Compare the text to each value in the CSV file
                # Parameter for our model: what is number of edits? 3? 4? will this catch too many?
                for csv_value in csv_values:
                    if levenshtein_distance(text, csv_value) <= self.editDist:
                        validated_orgs[text] = 1
                        break  # Avoid duplicates if text matches multiple CSV values
                    else:
                        # For orgs not in DB, want to additionally check that they meet some keyword requirements
                        if bool(re.search(pattern, text, re.IGNORECASE)):
                            validated_orgs[text] = 1

        return validated_orgs

    def processUtterance(self, utterance):
        # Return a dictionary that maps org --> # of occurences in the utterance
        return self.validate_org_entities(self.merge_org_entities(self.createTaggedOutputs(utterance)), "allOrgs.csv")


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


class NLTK_Model():
    def __init__(self, editDist):
        self.editDist = editDist

    def extract_named_entities(self, text, entity_type="ORGANIZATION"):
        """
        Extract named entities of a specific type from the text using NLTK's MaxEnt NER.
        """
        tokens = word_tokenize(text)  # Tokenize the text
        pos_tags = pos_tag(tokens)    # Get part-of-speech tags
        ner_tree = ne_chunk(pos_tags)  # Perform Named Entity Recognition

        entities = []
        for subtree in ner_tree:
            if isinstance(subtree, Tree) and subtree.label() == entity_type:
                entity_name = " ".join(
                    [token for token, pos in subtree.leaves()])
                entities.append(entity_name)
        return entities

    def validate_orgs(self, extracted_orgs):
        csv_values = []
        validated_orgs = dict()
        org_keywords = ['LLC', 'Inc.', 'Corp.', 'Limited', 'Ltd.', 'Co.', 'Foundation',
                        'Association', 'Institute', 'University', 'College', 'Group', 'Enterprise', 'Holding',
                        'Ventures', 'Partners']
        # Create a regex pattern to match any of the keywords, case-insensitive
        pattern = r'\b(?:' + '|'.join(re.escape(keyword)
                                      for keyword in org_keywords) + r')\b'
        with open("allOrgs.csv", "r", encoding="utf-8") as file:
            # Use DictReader to handle column names
            reader = csv.DictReader(file)
            for row in reader:
                # Replace 'org_names' with your column name
                csv_values.append(row["name"])
        for org in extracted_orgs:
            # Check against all values in the CSV
            for csv_value in csv_values:
                if levenshtein_distance(org, csv_value) <= self.editDist:
                    validated_orgs[org] = 1  # Mark as validated
                    break  # Stop further checks for this org
            else:
                # If no close match in CSV, validate with regex pattern
                if re.search(pattern, org, re.IGNORECASE):
                    validated_orgs[org] = 1  # Mark as validated by regex

        return validated_orgs

# Function to calculate F1 score


def f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def calcMetricsBert(bert, fileName):
    # Open the CSV file
    with open(fileName, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)  # Read the CSV as a dictionary

        # Variables to accumulate the metrics
        all_precision = []
        all_recall = []
        num_samples = 0
        # Loop over each row
        for row in csv_reader:
            utterance = row['utterance']
            validOrgs = bert.validate_org_entities(bert.merge_org_entities(
                bert.createTaggedOutputs(utterance)), "allOrgs.csv")
            # Convert validOrgs dictionary keys to a set of strings for comparison
            validOrgsSet = set(validOrgs.keys()) if validOrgs else set()

            orgs_labeled = literal_eval(row['orgs_labeled'])
            labeledOrgs = set()
            for org in orgs_labeled:
                # to avoid string issues with apostrophes inside ORG names
                labeledOrgs.add(org.replace('@', "'"))
            print(f'Labeled orgs: {labeledOrgs}')
            # Calculate precision and recall for this sample
            if len(validOrgsSet) == 0 and len(labeledOrgs) == 1 and '' in labeledOrgs:
                # If both sets empty, no true/false negatives/positives to add
                continue

            true_positive = len(labeledOrgs.intersection(validOrgsSet))
            false_positive = len(validOrgsSet) - true_positive
            false_negative = len(labeledOrgs) - true_positive

            precision = true_positive / \
                (true_positive + false_positive) if (true_positive +
                                                     false_positive) > 0 else 0
            recall = true_positive / \
                (true_positive + false_negative) if (true_positive +
                                                     false_negative) > 0 else 0

            # Store the precision and recall for macro and micro averaging
            all_precision.append(precision)
            all_recall.append(recall)
            num_samples += 1

        # Compute macro and micro averages
        macro_precision = sum(all_precision) / \
            num_samples if num_samples > 0 else 0
        macro_recall = sum(all_recall) / num_samples if num_samples > 0 else 0
        micro_precision = sum([tp for tp in all_precision]) / (sum(
            [tp + fp for tp, fp in zip(all_precision, all_recall)]) if num_samples > 0 else 1)
        micro_recall = sum([tp for tp in all_recall]) / (sum([tp + fn for tp,
                                                              fn in zip(all_precision, all_recall)]) if num_samples > 0 else 1)
        all_f1 = []

        for precision, recall in zip(all_precision, all_recall):
            f1 = f1_score(precision, recall)
            all_f1.append(f1)

        # Compute macro F1 score
        macro_f1 = mean(all_f1) if num_samples > 0 else 0

        # Compute micro F1 score
        micro_f1 = f1_score(micro_precision, micro_recall)

        # Specify the file path where you want to write the results
        output_file_path = 'base_model_test.txt'

        # Open the file for writing
        with open(output_file_path, 'a') as file:
            file.write(f"Edit distance to validate organizations: {
                       bert.editDist}\n")
            file.write(f"Macro Precision: {macro_precision:.2f}\n")
            file.write(f"Macro Recall: {macro_recall:.2f}\n")
            file.write(f"Micro Precision: {micro_precision:.2f}\n")
            file.write(f"Micro Recall: {micro_recall:.2f}\n")
            file.write(f"Macro F1 Score: {macro_f1:.2f}\n")
            file.write(f"Micro F1 Score: {micro_f1:.2f}\n")


def calcMetricsNLTK(fileName):
    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    # Open the CSV file
    nltkModel = NLTK_Model(3)
    with open(fileName, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)  # Read the CSV as a dictionary

        # Variables to accumulate the metrics
        all_precision = []
        all_recall = []
        num_samples = 0
        # Loop over each row
        for row in csv_reader:
            utterance = row['utterance']
            orgsNLTK = nltkModel.validate_orgs(
                nltkModel.extract_named_entities(utterance))
            validOrgsSet = set(orgsNLTK.keys()) if orgsNLTK else set()

            print(f'nltk orgs: {orgsNLTK}')
            orgs_labeled = literal_eval(row['orgs_labeled'])
            labeledOrgs = set()
            for org in orgs_labeled:
                # to avoid string issues with apostrophes inside ORG names
                labeledOrgs.add(org.replace('@', "'"))
            print(f'Labeled orgs: {labeledOrgs}')
            # Calculate precision and recall for this sample
            if len(validOrgsSet) == 0 and len(labeledOrgs) == 1 and '' in labeledOrgs:
                # If both sets empty, no true/false negatives/positives to add
                continue

            true_positive = len(labeledOrgs.intersection(validOrgsSet))
            false_positive = len(validOrgsSet) - true_positive
            false_negative = len(labeledOrgs) - true_positive

            precision = true_positive / \
                (true_positive + false_positive) if (true_positive +
                                                     false_positive) > 0 else 0
            recall = true_positive / \
                (true_positive + false_negative) if (true_positive +
                                                     false_negative) > 0 else 0

            # Store the precision and recall for macro and micro averaging
            all_precision.append(precision)
            all_recall.append(recall)
            num_samples += 1

        # Compute macro and micro averages
        macro_precision = sum(all_precision) / \
            num_samples if num_samples > 0 else 0
        macro_recall = sum(all_recall) / num_samples if num_samples > 0 else 0
        micro_precision = sum([tp for tp in all_precision]) / (sum(
            [tp + fp for tp, fp in zip(all_precision, all_recall)]) if num_samples > 0 else 1)
        micro_recall = sum([tp for tp in all_recall]) / (sum([tp + fn for tp,
                                                              fn in zip(all_precision, all_recall)]) if num_samples > 0 else 1)
        all_f1 = []

        for precision, recall in zip(all_precision, all_recall):
            f1 = f1_score(precision, recall)
            all_f1.append(f1)

        # Compute macro F1 score
        macro_f1 = mean(all_f1) if num_samples > 0 else 0

        # Compute micro F1 score
        micro_f1 = f1_score(micro_precision, micro_recall)

        # Specify the file path where you want to write the results
        output_file_path = 'nltk_model_test.txt'

        # Open the file for writing
        with open(output_file_path, 'a') as file:
            file.write(f"Macro Precision: {macro_precision:.2f}\n")
            file.write(f"Macro Recall: {macro_recall:.2f}\n")
            file.write(f"Micro Precision: {micro_precision:.2f}\n")
            file.write(f"Micro Recall: {micro_recall:.2f}\n")
            file.write(f"Macro F1 Score: {macro_f1:.2f}\n")
            file.write(f"Micro F1 Score: {micro_f1:.2f}\n")


def main():
    # Creating model, setting parameter for Levenstein distance

    bert = bertModel(3, model=AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER"), tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER"))

    # bert = bertModel(1, model=AutoModelForTokenClassification.from_pretrained('./fine_tuned_model'),
    #               tokenizer=AutoTokenizer.from_pretrained('./fine_tuned_model'))
    file_name = 'sample_utterances_test.csv'
    calcMetricsBert(bert, file_name)
    calcMetricsNLTK(file_name)


# Call the main function
if __name__ == "__main__":
    main()
