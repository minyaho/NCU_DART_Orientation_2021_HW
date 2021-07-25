import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Reshape dataset
x_train = x_train.reshape(len(x_train),784).astype('float32')
x_test = x_test.reshape(len(x_test),784).astype('float32')

# Normalize
x_train = x_train/255
x_test = x_test/255

# One-Hot Encoding
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

# Buliding Model
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=128, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training Model
train_history = model.fit(x = x_train, y = y_train_onehot, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

# Accuracy score
scores = model.evaluate(x_test, y_test_onehot)
print('\naccuracy: {}'.format(scores[1])) 