import pickle
import numpy as np
from keras.preprocessing.text import tokenizer
import json
from keras_preprocessing.sequence import pad_sequences

# # Load the tokenizer from the saved file
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)



# Convert the tokenizer to a dictionary
tokenizer_dict = tokenizer.to_json()

# Save the dictionary to a JSON file
with open('tokenizer.json', 'w') as handle:
    json.dump(tokenizer_dict, handle)

# Define the maximum sequence length
max_len = 165

# Load the trained model from the saved file
model = pickle.load(open('finalized_model.h5', 'rb'))

# Define a new input text sequence to predict the sentiment label for
new_text = 'This is a positive text'

# Tokenize and pad the new input text sequence
new_seq = tokenizer.texts_to_sequences([new_text])
new_seq = pad_sequences(new_seq, maxlen=max_len)

# Make a prediction using the trained model
pred = model.predict(new_seq)

# Print the predicted sentiment label
if np.argmax(pred) == 0:
    print('Negative')
else:
    print('Positive')
