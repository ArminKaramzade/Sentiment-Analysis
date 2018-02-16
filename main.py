import preprocess, read, feature_extraction, classification
import numpy as np

NUM_OF_AMAZON_TEXT = 200*1000
TRAIN = .75
AMAZON_PATH = 'amazon/reviews_Movies_and_TV_5.json'
AMAZON_TXT_PATH = 'amazon/reviews_Movies_and_TV_5.txt'
AMAZON_NORMALIZED_PATH = 'amazon/normalized_reviews_Movies_and_TV_5.txt'
SST_DATA_PATH = 'sst/data.txt'
SST_NORMALIZED_DATA_PATH = 'sst/normalized_data.txt'
SST_LABEL_PATH = 'sst/label.txt'

try:
    f = open(SST_NORMALIZED_DATA_PATH, 'r')
except FileNotFoundError:
    print('preprocessing sst data ...')
    texts = read.read_sst_data(SST_DATA_PATH)
    preprocess.preprocess(SST_NORMALIZED_DATA_PATH, texts)
    f = open(SST_NORMALIZED_DATA_PATH, 'r')
normalized_texts = f.readlines()

try:
    f = open(AMAZON_NORMALIZED_PATH, 'r')
except FileNotFoundError:
    try:
        f = open(AMAZON_TXT_PATH, 'r')
    except FileNotFoundError:
        read.json_to_text(AMAZON_PATH)
    print('preprocessing amazon data ...')
    texts = read.read_amazon_data(AMAZON_TXT_PATH, NUM_OF_AMAZON_TEXT)
    print(len(texts))
    preprocess.preprocess(AMAZON_NORMALIZED_PATH, texts)
    f = open(AMAZON_NORMALIZED_PATH, 'r')
normalized_train_model_texts = f.readlines()

y = read.read_sst_label(SST_LABEL_PATH)
split_point = int(TRAIN * len(y))
idx = np.random.permutation(y.shape[0])

# Tf-Idf features
print("tf_idf:")
X = feature_extraction._tfidf(normalized_texts)
X_train, y_train = X[:split_point], y[:split_point]
X_test, y_test = X[split_point:], y[split_point:]
classification.classify(X_train, y_train, X_test, y_test)

# LDA features
print("LDA:")
X = feature_extraction._lda(normalized_train_model_texts, normalized_texts, 100)
X_train, y_train = X[:split_point], y[:split_point]
X_test, y_test = X[split_point:], y[split_point:]
classification.classify(X_train, y_train, X_test, y_test)

# Doc2Vec features
print("doc2vec:")
X = feature_extraction._doc2vec(normalized_train_model_texts, normalized_texts, 1)
X_train, y_train = X[:split_point], y[:split_point]
X_test, y_test = X[split_point:], y[split_point:]
classification.classify(X_train, y_train, X_test, y_test)
