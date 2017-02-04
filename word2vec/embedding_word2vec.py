from math import sqrt
from operator import itemgetter
from random import randint

import numpy as np
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

from gensim.models import word2vec


def avg_word2vec(model, dataset='data/snli.test'):
    array_sentences = []
    array_embeddings = []
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            cont = 0
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.wv.vocab:
                    cont += 1
                    if avgword2vec is None:
                        avgword2vec = model[word]
                    else:
                        avgword2vec = avgword2vec + model[word]
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                avgword2vec = avgword2vec / cont  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'avg_word2vec: Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings


def cosine_similarity(a, b):
    assert len(a) == len(b), 'vectors need to have the same size'
    cos_sim = a.dot(b) / sqrt(a.dot(a)) / sqrt(b.dot(b))
    return cos_sim


def most_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])
    closest_idx = max(list_scores, key=list_scores.get)

    print 'The query :\n'
    print query_sentence + '\n'
    print 'is most similar to\n'
    print array_sentences[closest_idx]
    print 'with a score of : {0}\n'.format(list_scores[closest_idx])

    print '5 most similar sentences:'
    closest_5 = sorted(list_scores.iteritems(), key=itemgetter(1), reverse=True)[:5]
    for i, score in closest_5:
        print array_sentences[i], score

    return closest_idx

def most_5_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])

    closest_5 = sorted(list_scores.iteritems(), key=itemgetter(1), reverse=True)[:5]
    closest_5_idx = [i for i, score in closest_5]

    assert len(closest_5_idx) == 5

    return closest_5_idx


def IDF(dataset='data/snli.test'):
    # Compute IDF (Inverse Document Frequency). Here a "document" is a sentence.
    # word2idf['peach'] = IDF(peach)
    df = {}
    N = 0
    with open(dataset) as f:
        for line in f:
            N += 1
            sentence = line.split()
            sentence = np.unique(sentence)
            for word in sentence:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1

    word2idf = {}
    for k,v in df.iteritems():
        word2idf[k] = np.log(float(N) / v)

    return word2idf

def avg_word2vec_idf(model, word2idf, dataset='data/snli.test'):
    array_sentences = []
    array_embeddings = []
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            sumidf = 0
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.wv.vocab:
                    sumidf += word2idf[word]
                    if avgword2vec is None:
                        avgword2vec = word2idf[word] * model[word]
                    else:
                        avgword2vec = avgword2vec + word2idf[word] * model[word]
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                avgword2vec = avgword2vec / sumidf  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'avg_word2vec_idf: Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings

if __name__ == "__main__":

    if False: # FIRST PART
        sentences = word2vec.Text8Corpus('data/text8')

        # Train a word2vec model
        embedding_size = 200
        model = word2vec.Word2Vec(sentences, size=embedding_size)

        # Train a word2vec model with phrases
        bigram_transformer = gensim.models.Phrases(sentences)
        model_phrase = Word2Vec(bigram_transformer[sentences], size=200)
    else:
        # Loading model trained on words
        model = word2vec.Word2Vec.load('models/text8.model')

        # Loading model enhanced with phrases (2-grams)
        model_phrase = word2vec.Word2Vec.load('models/text8.phrase.model')

    """
    SECOND PART: Investigating word2vec word embeddings space
    """

    # Words that are similar are close in the sense of the cosine similarity.
    sim = model.similarity('woman', 'man')
    print 'Printing word similarity between "woman" and "man" : {0}'.format(sim)

    sim = model.similarity('apple', 'mac')
    print 'Printing word similarity between "apple" and "mac" : {0}'.format(sim)

    sim = model.similarity('apple', 'peach')
    print 'Printing word similarity between "apple" and "peach" : {0}'.format(sim)

    sim = model.similarity('banana', 'peach')
    print 'Printing word similarity between "banana" and "peach" : {0}'.format(sim)

    # And words that appear in the same context have similar word embeddings.
    print model.most_similar(['paris'])[0]
    print model_phrase.most_similar(['paris'])[0]

    print model.most_similar(['difficult'])
    print model_phrase.most_similar(['difficult'])

    print model_phrase.most_similar(['clinton'])[:3]

    # Compositionality and structure in word2vec space
    print model.most_similar(positive=['woman', 'king'], negative=['man'])[0]

    print model.most_similar(positive=['france', 'berlin'], negative=['germany'])[0]

    """
    THIRD PART: Sentence embeddings with average(word2vec)
    """
    data_path = 'data/snli.test'
    array_sentences, array_embeddings = avg_word2vec(model, dataset=data_path)

    query_idx =  777 # random sentence
    assert query_idx < len(array_sentences) # little check

    # array_sentences[closest_idx] will be the closest sentence to array_sentences[query_idx].
    closest_idx = most_similar(query_idx, array_embeddings, array_sentences)

    closest_5_idx = most_5_similar(query_idx, array_embeddings, array_sentences)

    print 'Most 5 similar:\n'
    for idx in closest_5_idx:
        print array_sentences[idx]

    """
    FOURTH PART: Weighted average of word vectors with IDF.
    """
    word2idf = IDF(data_path)

    words = ['the', 'a' , 'clinton', 'woman', 'man', 'apple', 'peach', 'banana', 'mac', 'paris', 'france']

    for word in words:
        if word in word2idf:
            print word, word2idf[word]
        else:
            print word, "not found"

    array_sentences_idf, array_embeddings_idf = avg_word2vec_idf(model, word2idf, dataset=data_path)
    closest_idx_idf = most_similar(query_idx, array_embeddings_idf, array_sentences_idf)
