from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict

def splitData(texts, labels, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    texts, labels = np.array(texts), np.array(labels)
    label_to_indices = defaultdict(list)

    # Group indices by label
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    train_indices, dev_indices = [], []

    # Stratified splitting
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)  # Shuffle within class
        split_idx = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split_idx])
        dev_indices.extend(indices[split_idx:])

    # Shuffle final sets to remove any ordering bias
    np.random.shuffle(train_indices)
    np.random.shuffle(dev_indices)

    return (
        texts[train_indices].tolist(), texts[dev_indices].tolist(),
        labels[train_indices].tolist(), labels[dev_indices].tolist()
    )

def vectorizeText(text, vocabulary, vocabulary_set):
    words_in_text = set(word_tokenize(text.lower())) & vocabulary_set  
    vector = [1 if word in words_in_text else 0 for word in vocabulary]
    
    return vector

def vectorizeTexts(texts, vocabulary):
    texts_vector = []
    vocabulary_set = set(vocabulary)
    for text in texts:
        texts_vector.append(vectorizeText(text, vocabulary, vocabulary_set))
    return texts_vector


# def vectorizeText(text, vocab_dict):
#     words_in_text = set(word_tokenize(text.lower()))
#     vector = np.zeros(len(vocab_dict), dtype=np.uint8)

#     for word in words_in_text:
#         if word in vocab_dict:  # O(1) lookup
#             vector[vocab_dict[word]] = 1

#     return vector

# def vectorizeTexts(texts, vocabulary):
#     vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}  # Dictionary for fast lookups
#     vectors = np.zeros((len(texts), len(vocabulary)), dtype=np.uint8)

#     for i, text in enumerate(texts):
#         vectors[i] = vectorizeText(text, vocab_dict)

#     return vectors