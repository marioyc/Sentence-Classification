from __future__ import print_function

##########################
## Importing packages
##########################
import numpy as np
np.random.seed(1234)  # for reproducibility (manually setting random seed)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D
from utils import load_imdb

import matplotlib.pyplot as plt

##########################
## Preparing data
##########################
vocab_size   =  5000 # number of words considered in the vocabulary
train_split = 0.7     # ratio of train sentences

(X_train, y_train), (X_test, y_test) = load_imdb(nb_words=vocab_size, train_split=train_split)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

## Padding input data
maxlen  = 80  # cut texts after this number of words (among top vocab_size most common words)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test  = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

##########################
## Building model
##########################
embed_dim = 25
nhid      = 128
nb_filter = 250
filter_length = 3
hidden_dims = 250
print('\nBuilding model...')

model = Sequential()
model.add(Embedding(vocab_size,
                    embed_dim,
                    input_length=maxlen,
                    dropout=0.2))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(LSTM(nhid, dropout_W=0.2, dropout_U=0.2))

model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
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
batch_size       =  64  # size of the minibach (each batch will contain 32 sentences)
nb_epoch         =  7

print('\n\nStarting training of the model\n')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(range(1,nb_epoch + 1), history.history['loss'], 'b', range(1,nb_epoch + 1), history.history['val_loss'], 'r')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(range(1,nb_epoch + 1), history.history['acc'], 'b', range(1,nb_epoch + 1), history.history['val_acc'], 'r')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

##########################
## Evaluate on test set
##########################
# evaluate model on test set (never seen during training)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('\n\nTest score:', score)
print('Test accuracy:', acc)
