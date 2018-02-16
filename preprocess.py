import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess(filename, texts):
    wordnet_lemmatizer = WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # remove punctuation, digits and stopwords & to lower
    new_texts = [[w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stopwords] for text in texts]
    # verbs to origin
    new_texts = [[wordnet_lemmatizer.lemmatize(w, 'v') if pos[0] == 'V' else w for w, pos in nltk.pos_tag(text)] for text in new_texts]
    f = open(filename, 'w')
    f.write(' '.join(new_texts[0]))
    for i in range(1, len(new_texts)):
        f.write('\n')
        f.write(' '.join(new_texts[i]))
    f.close()
