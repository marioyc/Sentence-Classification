import numpy as np
from keras.datasets import imdb


def load_imdb(nb_words, train_split=0.8):
    print 'Preparing IMDB-review sentence classification dataset with {0} % training data ...'.format(train_split*100)
    (X_1, y_1), (X_2, y_2) = imdb.load_data(nb_words=nb_words)
    X = np.array([x for x in X_1] + [x for x in X_2])
    Y = np.array([y for y in y_1] + [y for y in y_2])
    X_train, y_train = X[:int(train_split * len(X))], Y[:int(train_split * len(Y))]
    X_test, y_test = X[int(train_split * len(X)):], Y[int(train_split * len(Y)):]

    return (X_train, y_train), (X_test, y_test)