from torchtext.datasets import IMDB
from collections import defaultdict
from nltk.tokenize import word_tokenize
import time

import math

def sanitize(n, low_threshold, input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    if (n):
        lines = lines[n:]

    if (low_threshold):
        result_lines = []
        for line in lines:
            try:
                word, number = line.rsplit(",", 1)
                if float(number.strip()) >= low_threshold:
                    result_lines.append(f"{word}\n")
            except ValueError:
                # Skip lines that are not properly formatted
                print('err: ',word, number )

    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(result_lines)

    print(f"Processed file saved to {output_file}")


def writeResult(words_IG):
    with open("IG.txt", 'w', encoding="utf-8") as file:
        for word, IG in words_IG:
            file.write(f"{word}, {IG}\n")

def getAppearances(word, texts_array):
    x = 1
    count = 0
    for array in texts_array[1]:
        if array[word] == x:
            count+=1
    for array in texts_array[2]:
        if array[word] == x:
            count+=1
    return count

def getP_C_X(texts_array, c, x, word, N):
    # The probability that C is c, given that X is x
    # a Get num of texts that X=x 
    # b find how many of them are C=x
    #  b / a
    # not implemented yet

    count = 0
    for array in texts_array[c]:
        if array[word] == x:
            count+=1

    return count /  N

def getHC(x, word, N, texts_array):
    # Η εντροπία του C δεδομένου ότι το X είναι x
    p1 = getP_C_X(texts_array, c=1, x=x, word=word, N=N)
    p2 = getP_C_X(texts_array, c=2, x=x, word=word, N=N)

    if p1==0 and p2==0:
        return 0
    if p1==0:
        return -p2*math.log(p2,2)
    if p2==0:
        return -p1*math.log(p1,2)


    return -p1*math.log(p1,2) -p2*math.log(p2,2)

def getIG(word, texts_array, HC_generall, all_texts_number):
    all_appearances = getAppearances(word, texts_array)
    all_not_appearances = all_texts_number - all_appearances
    P_X_0 = all_not_appearances / all_texts_number
    P_X_1 = all_appearances / all_texts_number
    if P_X_1 == 0:
        return 0

    return HC_generall - P_X_0 * getHC(0, word, all_not_appearances, texts_array) - P_X_1 * getHC(1, word, all_appearances, texts_array)


def writeIGs(vocabulary, texts_array, HC_generall, all_texts_number):
    words_IG = []
    for X in range(0,len(vocabulary)):
        IG = getIG(X, texts_array, HC_generall, all_texts_number)
        words_IG.append((vocabulary[X], IG))
        
    words_IG.sort(key=lambda x: x[1], reverse=True)

    writeResult(words_IG)

def text_to_vector(text, vocabulary, vocabulary_set):
    words_in_text = set(word_tokenize(text.lower())) & vocabulary_set  
    vector = [1 if word in words_in_text else 0 for word in vocabulary]
    return vector

def dataToArray(vocabulary, train_data):
    # neg = 1, pos = 2
    texts_array = {
        1:[

        ],
        2:[

        ]
    }

    start = time.time()
    vocabulary_set = set(vocabulary)

    for label, text in train_data:
        texts_array[label].append(text_to_vector(text, vocabulary, vocabulary_set))

    print(f"Texts to array transformed: {time.time() - start}")
    return texts_array

# read initial vocabulary to Array / to tuple?
# Load vocabulary
def getInitialVocabularyArray():
    with open("imdb.vocab", "r", encoding="utf-8") as vocab_file:
        vocabulary =  vocab_file.read().splitlines()
        return vocabulary

def initializeVocWordCount(text, word_frequencies, vocabulary):
    text = text.lower()
    words_in_file = set(word_tokenize(text)) & vocabulary  # Intersection for performance
    for word in words_in_file:
            word_frequencies[word] += 1
    
def writeInitialAppearances(sorted_word_frequencies):
    # Open the file in write mode
    with open('vocab_result.txt', 'w', encoding="utf-8") as file:    
        # Write each word and its frequency to the file
        for word, frequency in sorted_word_frequencies:
            file.write(f"{word}, {frequency}\n")

def countInitiavlAppearances():
    # Load the vocabulary from a file (same as before)
    with open("imdb.vocab", "r", encoding="utf-8") as vocab_file:
        vocabulary = set(vocab_file.read().splitlines())  # Vocabulary as a set

    word_frequencies = defaultdict(int)  # For word frequency counting

    # Load the IMDB dataset
    train_data = IMDB(split="train")  # Load only the training data
    texts = [text for _, text in train_data]
    for text in texts:
        initializeVocWordCount(text, word_frequencies, vocabulary)

    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_frequencies

# load train data, load developemtn data 
def loadTrainAndDevelopmentData(splitter):

    train_data = list(IMDB(split="train"))  # Convert to list (label, text)
    half_index = len(train_data) // 2
    neg = train_data[:half_index]
    pos = train_data[half_index:]

    split_index = int(len(neg) * splitter)

    # Split positive data
    pos_train = pos[:split_index]
    pos_dev = pos[split_index:]

    neg_train = neg[:split_index]
    neg_dev = neg[split_index:]

    # train and dev

    return pos_train, pos_dev, neg_train, neg_dev


# create texts array 
def textToArray(text, vocabulary):
    words_in_text = set(text.split()) & set(vocabulary)
    vector = [1 if word in words_in_text else 0 for word in vocabulary]
    return vector

def textsToArray(texts, vocabulary):
    texts_arrays = {
        1:[

        ],
        2:[

        ]
    }
    for tuple in (texts) :
        text = tuple[1]
        label = tuple[0]
        text_vector = textToArray(text, vocabulary)
        texts_arrays[label].append(text_vector)
    return texts_arrays


# count appearances of vocabulary 
def countVocabularyAppearances(texts_arrays, vocabulary):
    vocabulary_appearances = [] # (word, count)
    for word in vocabulary:
        vocabulary_appearances.append([word, 0])
    
    for array in texts_arrays[1]:
        for word_index in array:
            vocabulary_appearances[word_index][1] += 1
    for array in texts_arrays[2]:
        for word_index in array:
            vocabulary_appearances[word_index][1] += 1

    return vocabulary_appearances


# sort vocabulary by appearances
def sortVocabularyByAppearances(vocabulary_appearances):
    sorted_vocabulary = sorted(vocabulary_appearances, key=lambda x: x[1])
    return sorted_vocabulary

# write voabuary appearances to file
def writeVocabularyAppearancesToFile(sorted_vocabulary):
    with open("vocabulary_appearances.txt", "w", encoding="utf-8") as file:
        for v in sorted_vocabulary:
            file.write(f"{v[0]}, {v[1]}\n")


# modify


# # calculate IG 
# def calculateIG(vocabulary, pos_train, neg_train):

#     number_of_texts_per_category = len(pos_train) + len(neg_train)
#     vocabulary_appearances = countVocabularyAppearances((pos_train+neg_train), vocabulary)
#     sorted_vocabulary = sortVocabularyByAppearances(vocabulary_appearances)
#     writeVocabularyAppearancesToFile(sorted_vocabulary)

#     IG = {}
#     for word in vocabulary:
#         count_0 = vocabulary_appearances[word] - count_1
#         count_1 = number_of_texts_per_category - count_0

#         result_0 = (count_0+1)/(number_of_texts_per_category+2)
#         result_1 = (count_1+1)/(number_of_texts_per_category+2)
#         IG[word] = result_0, result_1

# write IG
    # with open("IG.txt", "w", encoding="utf-8") as file:
    #     for word, (result_0, result_1) in IG.items():
    #         file.write(f"{word}, {result_0}, {result_1}\n")

    # return IG

# sanitize IG vocabulary

# read vocabulary from IG

# calc naive baytes
    # store P(X=x|C=c) for every word in the vocabulary
    # do the algorithm for all texts 
    # check results and play with hyper parameters

# presision , recall f1 ..

# graphs

# compare

# scitkit learn



