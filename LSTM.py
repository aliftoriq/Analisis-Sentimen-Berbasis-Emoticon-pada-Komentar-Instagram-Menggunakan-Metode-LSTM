import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('master_emoji.csv')

print(df)


# Define the number of features and the maximum length of the sequences
max_features = 2000
max_len = 165

# Tokenize the text
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['Special Tag'].values)
X = tokenizer.texts_to_sequences(df['Special Tag'].values)
X = pad_sequences(X, maxlen=max_len)

# Save the tokenizer object to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def map_sentiment(sentiment):
    if sentiment == 'negative':
        return 0
    elif sentiment == 'positive':
        return 1
    else:
        raise ValueError('Unknown sentiment: {}'.format(sentiment))

df['Sentiment'] = df['Sentiment'].apply(map_sentiment)


# Split the data into training and testing sets
# Y = pd.get_dummies(df['Sentiment']).values
Y = to_categorical(df['Sentiment'].values) # convert to one-hot encoded format
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Y_train = to_categorical(Y_train, 3)
# Y_test = to_categorical(Y_test, 3)

# Define the LSTM model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax')) # change from 2 to 3 units


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Y_train = tf.one_hot(Y_train, 3)
# Train the model
batch_size = 5
history = model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, Y_test))

# Plot the training and validation accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# Evaluate the model
score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)



filename = 'finalized_model.h5'
pickle.dump(model, open(filename, 'wb'))
