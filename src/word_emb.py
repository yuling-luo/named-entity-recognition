import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import gensim
from gensim.models import KeyedVectors
import nltk


def readfile(filename):
    """

    :param filename: file
    :return: [['word','entity']] format
    Reference: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs
    """
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-docstart') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


data = readfile("/Users/yulingluo/Documents/named-entity-recognition-conll2003/dataset/train.csv")

# read csv
train = pd.read_csv("/Users/yulingluo/Documents/named-entity-recognition-conll2003/dataset/train.csv")

# train test split
X_train = train['Sentence'].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(X_train))
words = tokenizer.word_index.items()


def glove_word_embeddings(doc_words):
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

    size_of_vocab = len(doc_words) + 1

    glove_embeddings = np.zeros((size_of_vocab, 100))  # use glove 100 dimensions embeddings

    for words, i in doc_words:
        if words in word:
            embedding_vector = embeddings_index.get(word)
            glove_embeddings[i] = embedding_vector
        else:
            embedding_vector = np.random.uniform(-0.25, 0.25, 100)  # for unknown token
            glove_embeddings[i] = embedding_vector

    return glove_embeddings[1:]  # because the first one is all 0 due to different ways to start index from word dict


glove_embeddings = glove_word_embeddings(words)


def get_w2v_embeddings(doc_words):
    """

    :param doc: doc
    :return: w2v embeddings
    """
    w2v = KeyedVectors.load_word2vec_format(
        '/Users/yulingluo/Documents/named-entity-recognition-conll2003/embeddings/GoogleNews-vectors-negative300.bin',
        binary=True)
    size_of_vocab = len(doc_words) + 1

    w2v_embeddings = np.zeros((size_of_vocab, w2v.vector_size))

    for word, i in doc_words:
        try:
            w2v_vector = w2v[word]
        except:
            w2v_vector = np.random.uniform(-0.25, 0.25, 300)  # for unknown tokens
        w2v_embeddings[i] = w2v_vector

    return w2v_embeddings[1:]


w2v_embeddings = get_w2v_embeddings(words)

# extract word_list
word_list = [k[0] for k in words]


def get_char_embeddings(words_list):
    """

    :param doc_words: words
    :return: character embedding for words
    """
    maxlen = len(max(words_list, key=len))

    char2Idx = dict()
    for c in " 0123456789abcdefghijklmnopqrstuvwxyz.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)

    char_emb = []
    for i in range(len(words_list)):
        input_data = []
        for c in words_list[i]:
            integer = [char2Idx[c]]
            input_data.append(integer)

        emb = np.zeros((maxlen, len(char2Idx)), int)  # standardize size
        for idx, n in enumerate(input_data):
            emb[idx][n] = 1

        char_emb.append(emb)

    return char_emb


char_embeddings = get_char_embeddings(word_list)


def pos_tag_generate(doc_words):
    """

    :param doc_words: words in doc
    :return: pos tags
    """
    pos_tag = nltk.pos_tag(doc_words)

    return pos_tag


pos_tags = pos_tag_generate(word_list)

