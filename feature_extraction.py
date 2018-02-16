import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models import doc2vec

def _tfidf(data):
    tf = TfidfVectorizer(max_df=1., min_df=7, max_features=None, binary=True)
    return tf.fit_transform(data)


def _lda(train_model_data, data, n_topics):
    tf_vectorizer = CountVectorizer(max_df=1., min_df=1, max_features=None, binary=True)
    tf_train_model = tf_vectorizer.fit_transform(train_model_data)
    tf_data = tf_vectorizer.transform(data)
    lda = LDA(n_components=n_topics)
    lda.fit(tf_train_model)
    return lda.transform(tf_data)

def _doc2vec(train_model_data, data, try_to_load=False):
    def labelize(docs, label_type):
        labelized = []
        for i, doc in enumerate(docs):
            label = "%s_%s"%(label_type, i)
            labelized.append(doc2vec.LabeledSentence(doc, [label]))
        return labelized
    try:
        if try_to_load:
            model = doc2vec.Doc2Vec.load('model/doc2vec')
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        model = doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
        X = labelize(train_model_data, 'AMAZON')
        model.build_vocab(X)
        model.train(X, total_examples=model.corpus_count, epochs=20)
        model.save('model/doc2vec')
    ret = []
    for d in data:
        ret.append(model.infer_vector(d))
    return np.array(ret)
