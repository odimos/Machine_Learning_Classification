import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.datasets import IMDB
from sklearn.model_selection import train_test_split
import time 

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
import pandas as pd

from plots import plotResults

def log_time(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Read vocab.txt and convert it into a list
print("Reading vocabulary...")
with open("vocab.txt", "r", encoding="utf-8") as file:
    vocabulary = [line.strip() for line in file]

# Load IMDB dataset
log_time("Loading IMDB dataset...")
start_time = time.time()
train_data = list(IMDB(split="train"))  # (label, text)
log_time(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

log_time("Extracting texts and labels...")
texts = [text for label, text in train_data]
labels = [label for label, text in train_data]

# Split data (80% Train, 20% Dev)
log_time("Splitting dataset into train/dev...")
start_time = time.time()
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
log_time(f"Dataset split in {time.time() - start_time:.2f} seconds.")

#  Vectorize texts
log_time("Vectorizing train and dev sets...")
start_time = time.time()
vectorizer = CountVectorizer(vocabulary=vocabulary, binary=True)
train_vectors = vectorizer.transform(train_texts).toarray()
dev_vectors = vectorizer.transform(dev_texts).toarray()
log_time(f"Vectorization completed in {time.time() - start_time:.2f} seconds.")

def useModel(model, train_vectors, train_labels, dev_vectors, dev_labels):
    # Train
    model.fit(train_vectors, train_labels)
    # Predict in Train
    train_predictions = model.predict(train_vectors)
    # Predict in Dev
    dev_predictions = model.predict(dev_vectors)

    # Metrics train
    train_precision = precision_score(train_labels, train_predictions, pos_label=1)
    train_recall = recall_score(train_labels, train_predictions, pos_label=1)
    train_f1 = f1_score(train_labels, train_predictions, pos_label=1)

    # Metrics dev
    dev_precision  = precision_score(dev_labels, dev_predictions, pos_label=1)
    dev_recall  = recall_score(dev_labels, dev_predictions, pos_label=1)
    dev_f1  = f1_score(dev_labels, dev_predictions, pos_label=1)

    return train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1

print(f"Train vectors shape: {train_vectors.shape}")

nb_model = MultinomialNB()

train_size_percentages = np.linspace(0.1, 1.0, 10)

train_precisions, dev_precisions = [], []
train_recalls, dev_recalls = [], []
train_f1s, dev_f1s = [], []

for size in train_size_percentages:
    n_samples = int(size * len(train_labels))
    
    subset_vectors = train_vectors[:n_samples]
    subset_labels = train_labels[:n_samples]
    train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1 =  useModel(nb_model, subset_vectors, subset_labels, dev_vectors, dev_labels)

    train_precisions.append(train_precision)
    dev_precisions.append(dev_precision)
    train_recalls.append(train_recall)
    dev_recalls.append(dev_recall)
    train_f1s.append(train_f1)
    dev_f1s.append(dev_f1)

plotResults(train_size_percentages, train_labels, train_precisions, dev_precisions, train_recalls, dev_recalls, train_f1s, dev_f1s)

# second metrics

# Train the model using all the training data
combined_vectors = np.vstack([train_vectors, dev_vectors])
combined_labels = np.concatenate([train_labels, dev_labels]) 
nb_model.fit(combined_vectors, combined_labels)

# Load IMDB dataset tests
log_time("Loading IMDB dataset test...")
start_time = time.time()
test_data = list(IMDB(split="test"))  # (label, text)
log_time(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

# Separate the labels and texts
test_labels = [label for label, _ in test_data]
test_texts = [text for _, text in test_data]

# Vectorize the test texts (using the same vectorizer as before)
vectorizer = CountVectorizer(vocabulary=vocabulary, binary=True)
test_vectors = vectorizer.fit_transform(test_texts).toarray()
# Predict on the test data
test_predictions = nb_model.predict(test_vectors)

# Metrics (micro and macro)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average=None)  # None for per-class scores
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(test_labels, test_predictions, average='micro')
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, test_predictions, average='macro')

results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Class 0': [precision[0], recall[0], f1[0]],
    'Class 1': [precision[1], recall[1], f1[1]],
    'Micro-averaged': [precision_micro, recall_micro, f1_micro],
    'Macro-averaged': [precision_macro, recall_macro, f1_macro]
})
print(results_df)