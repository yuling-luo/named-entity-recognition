import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import gensim
from gensim.models import KeyedVectors

train = pd.read_csv("/Users/yulingluo/Documents/named-entity-recognition-conll2003/dataset/train.csv")

# train test split
X_train = train['Sentence'].values


def glove_word_embeddings(doc):
    """

    :param glove: pre-trained glove embeddings (glove 6b.100d)
    :return: word embeddings for train document
    Reference: https://www.analyticsvidhya.com/blog/2020/03/pretrained-word-embeddings-nlp/
    """
    embeddings_index = dict()
    f = open("/Users/yulingluo/Documents/named-entity-recognition-conll2003/embeddings/glove.6B.100d.txt")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(doc))
    size_of_vocab = len(tokenizer.word_index) + 1

    glove_embeddings = np.zeros((size_of_vocab, 100))  # use glove 100 dimensions embeddings

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            glove_embeddings[i] = embedding_vector

    return glove_embeddings[1:]  # because the first one is all 0 due to different ways to start index from word dict


glove_embeddings = glove_word_embeddings(X_train)


def get_w2v_embeddings(doc):
    """

    :param doc: doc
    :return: w2v embeddings
    """
    w2v = KeyedVectors.load_word2vec_format(
        '/Users/yulingluo/Documents/named-entity-recognition-conll2003/embeddings/GoogleNews-vectors-negative300.bin',
        binary=True)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(doc))
    size_of_vocab = len(tokenizer.word_index) + 1

    w2v_embeddings = np.zeros((size_of_vocab, w2v.vector_size))

    for word, i in tokenizer.word_index.items():
        try:
            w2v_vector = w2v[word]
        except:
            w2v_vector = 0  # assign words that not in w2v vocabulary as 0
        w2v_embeddings[i] = w2v_vector

    return w2v_embeddings[1:]


w2v_embeddings = get_w2v_embeddings(X_train)

