import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('scores.csv')

# Filter tags where predictions are True
tags = df[df['predictions'] == True]

# Extract scores and reason
scores = tags['score']
reasons = tags['reason']

# Define bins from 0 to 1 with increments of 0.1
bins = [i / 10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
labels = [f'{b:.1f} - {b + 0.1:.1f}' for b in bins[:-1]]  # Bin labels: '0.0 - 0.1', '0.1 - 0.2', etc.

# Group by 'reason' and plot for each group
for reason, group in tags.groupby('reason'):
    # Categorize scores into these bins for the current group
    binned_scores = pd.cut(group['score'], bins=bins, labels=labels, right=False)

    # Count the frequency of each bin
    score_counts = binned_scores.value_counts().sort_index()

    # Create a bar plot for each reason
    plt.figure(figsize=(10, 6))
    plt.bar(score_counts.index, score_counts.values, color='skyblue')
    plt.title(f'Frequency Distribution of Scores for Reason: {reason}', fontsize=16)
    plt.xlabel('Score Range', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()
