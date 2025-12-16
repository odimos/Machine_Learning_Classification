import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.datasets import IMDB
import time
import pandas as pd

from functions import splitData, vectorizeTexts 
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

from plots import plotResults


# Function to print timestamps
def log_time(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Read vocab.txt and convert it into a list
print("Reading vocabulary...")
with open("vocab.txt", "r", encoding="utf-8") as file:
    vocabulary = [line.strip() for line in file]

# 1. Load IMDB dataset
log_time("Loading IMDB dataset...")
start_time = time.time()
train_data = list(IMDB(split="train"))  # Converts to a list of (label, text)
log_time(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

# 2. Extract texts and labels
log_time("Extracting texts and labels...")
texts = [text for label, text in train_data]
labels = [label for label, text in train_data]

#####
# texts = texts[:1000]
# labels = labels[:1000]

train_texts, dev_texts, train_labels, dev_labels = splitData(texts, labels, test_size=0.2, random_state=42)

log_time("Vectorizing train and dev sets...")
train_vectors =  vectorizeTexts(train_texts, vocabulary)
dev_vectors = vectorizeTexts(dev_texts, vocabulary)
log_time(f"Vectorization completed in {time.time() - start_time:.2f} seconds.")


class NaiveBaytes:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def calculate_P_Xi_given_C_all(self, word, x, y):

        # Initialize counters
        count_c1 = 0
        count_c2 = 0
        
        # Count docs containing the word
        for doc, label in zip(x, y):
            if doc[word] == 1:  # if word is present
                if label == 1:
                    count_c1 += 1
                else:  # label == 2
                    count_c2 += 1
        
        # Calculate docs without the word
        docs_without_word_c1 = self.size - count_c1
        docs_without_word_c2 = self.size - count_c2
        
        # Calculate all probabilities with Laplace smoothing
        P_X_1__given_C_1 = (count_c1 + 1) / (self.size + 2)     # P(X=1|C=1)
        P_X_0__given_C_1 = (docs_without_word_c1 + 1) / (self.size + 2)  # P(X=0|C=1)
        P_X_1__given_C_2 = (count_c2 + 1) / (self.size + 2)     # P(X=1|C=2)
        P_X_0__given_C_2 = (docs_without_word_c2 + 1) / (self.size + 2)  # P(X=0|C=2)
        
        return P_X_1__given_C_1, P_X_0__given_C_1, P_X_1__given_C_2, P_X_0__given_C_2

    def fit(self, x,y):
        self.size = int (len(x)/2)
        self.vocabulary_calculated = {}

        for index,word in enumerate(vocabulary):
            P_X_1__given_C_1, P_X_0__given_C_1, P_X_1__given_C_2, P_X_0__given_C_2 = self.calculate_P_Xi_given_C_all(index,x, y)
            self.vocabulary_calculated[index] = {
                0: {
                    1:P_X_0__given_C_1, 2:P_X_0__given_C_2
                },
                1: {
                    1:P_X_1__given_C_1, 2:P_X_1__given_C_2
                }
            }

    def predictVector(self,vector):
        sum1 = (1/2) # P(C=1)
        sum2 = (1/2) # P(C=2)
        for index, word in enumerate(self.vocabulary):
            x = vector[index]
            sum1 *= self.vocabulary_calculated[index][x][1]
            sum2 *= self.vocabulary_calculated[index][x][2]
        
        if sum1 > sum2:return 1 
        return 2

    def predict(self, x):
        x_predictions = []
        for text in x:
            prediction = self.predictVector(text)
            x_predictions.append(prediction)
        return x_predictions
    

model = NaiveBaytes(vocabulary)

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

train_size_percentages = np.linspace(0.1, 1.0, 10)

train_precisions, dev_precisions = [], []
train_recalls, dev_recalls = [], []
train_f1s, dev_f1s = [], []

for size in train_size_percentages:
    n_samples = int(size * len(train_labels))
    
    subset_vectors = train_vectors[:n_samples]
    subset_labels = train_labels[:n_samples]
    train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1 = useModel(
        model, subset_vectors, subset_labels, dev_vectors, dev_labels)
    
    train_precisions.append(train_precision)
    dev_precisions.append(dev_precision)
    train_recalls.append(train_recall)
    dev_recalls.append(dev_recall)
    train_f1s.append(train_f1)
    dev_f1s.append(dev_f1)

plotResults(train_size_percentages, train_labels, train_precisions, dev_precisions, train_recalls, dev_recalls, train_f1s, dev_f1s)


# Train the model using all the training data
combined_vectors = np.vstack([train_vectors, dev_vectors])
combined_labels = np.concatenate([train_labels, dev_labels]) 
model.fit(combined_vectors, combined_labels)

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
test_predictions = model.predict(test_vectors)

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