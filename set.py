from functions2 import *
import math

def load_words(file_path):
    arr = []
    with open(file_path, 'r', encoding="utf-8") as f:
        arr = f.read().splitlines()
    return arr

def getInitialAppearancesSanitized(n, k_threshold):
    sorted_word_frequencies = countInitiavlAppearances()
    writeInitialAppearances(sorted_word_frequencies)
    sanitize(n, k_threshold, "vocab_result.txt", "vocab_result_sanitized.txt")


def generateVocabularyFromIGs(m_threshold):

    vocabulary = load_words("vocab_result_sanitized.txt")
    pos_train, pos_dev, neg_train, neg_dev = loadTrainAndDevelopmentData(0.8)
    train_data = neg_train + pos_train 

    neg_data_length = len(neg_train)
    pos_data_length = len(pos_train)
    all_data_length = neg_data_length + pos_data_length

    P_C_1 = neg_data_length/all_data_length
    P_C_2 = pos_data_length/all_data_length
    HC_generall = -P_C_1 * math.log(P_C_1, 2) -P_C_2 * math.log(P_C_2, 2)

    texts_array = dataToArray(vocabulary, train_data)
    writeIGs(vocabulary, texts_array, HC_generall, all_data_length)
    # threshold 0.01

    sanitize(False, m_threshold, "IG.txt", "vocab.txt")

# k low limit 34, n=100, m_threshold 0.0016
getInitialAppearancesSanitized(100, 30)
generateVocabularyFromIGs(0.0016)
