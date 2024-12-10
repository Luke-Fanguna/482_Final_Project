from transformers import Trainer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("leondz/wnut_17")

# Access the train and test splits
train_dataset = dataset['train']
test_dataset = dataset['test']
print(train_dataset)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Define a function to tokenize and align labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, padding='max_length', max_length=128, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        # Map tokenized inputs back to word IDs
        word_ids = tokenized_inputs.word_ids(i)
        label_ids = [-100 if word_id is None else label[word_id]
                     for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# Apply the tokenization and alignment function to the train and test splits
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# Check the type of the 'labels' feature to confirm
print(train_dataset.features['labels'])

# If the labels feature is not a ClassLabel, you can manually set num_labels like this:
num_labels = len(
    set([label for example in train_dataset for label in example['labels']]))
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", num_labels=num_labels, ignore_mismatched_sizes=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=8,  # Adjust as needed
    per_device_eval_batch_size=8,  # Adjust as needed
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Set up data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
