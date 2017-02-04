from __future__ import print_function
'''This example demonstrates the use of Convolution1D for text classification.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

'''
Simple Convolution1D for Sequence Classification
'''

##########################
## Importing packages
##########################
# importing packages/function that will be useful later
import numpy as np
np.random.seed(1234)  # for reproducibility (manually setting random seed)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D, GlobalMaxPooling1D
from utils import load_imdb

##########################
## Preparing data
##########################
# some parameters
vocab_size   =  5000 # number of words considered in the vocabulary
train_split = 0.7     # ratio of train sentences

# Preparing data is usually the most time-consuming part of machine learning.
# Luckily for you, the imdb dataset has already been preprocessed and included in Keras.
(X_train, y_train), (X_test, y_test) = load_imdb(nb_words=vocab_size, train_split=train_split)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

## Padding input data
# Models in Keras (and elsewhere) usually take as input batches of sentences of the same length.
# Since sentences usually have different sizes, we "pad" sentences (we add a dummy "padding" token at the end of the
# sentences. The input thus has this size : (batchsize, maxseqlen) where maxseqlen is the maximum length of a sentence
# in the batch.

maxlen  = 80  # cut texts after this number of words (among top vocab_size most common words)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test  = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

##########################
## Building model
##########################

embed_dim = 16
nhid      = 128
print('\nBuilding model...')

nb_filter = 250
filter_length = 3
hidden_dims = 250




model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(vocab_size,
                    embed_dim,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use temporal max pooling:
model.add(GlobalMaxPooling1D())

# We add a classifier (MLP with one hidden layer)
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


##########################
## Define (i)   loss function
#         (ii)  optimizer
#         (iii) metrics
##########################

loss_classif     =  'binary_crossentropy'
optimizer        =  'adam' # or sgd
metrics_classif  =  ['accuracy']

model.compile(loss=loss_classif,
              optimizer=optimizer,
              metrics=metrics_classif)

print(model.summary())
print('Built model')
##########################
## Train Model
##########################
validation_split =  0.2 # Held-out ("validation") data to test on.
batch_size       =  32  # size of the minibach (each batch will contain 32 sentences)

print('\n\nStarting training of the model\n')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=6, validation_split=0.2)

##########################
## Evaluate on test set
##########################
# evaluate model on test set (never seen during training)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('\n\nTest score:', score)
print('Test accuracy:', acc)
