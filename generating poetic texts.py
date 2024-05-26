import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Download the text data
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the text data
with open(filepath, 'rb') as file:
    text = file.read().decode(encoding='utf-8').lower()

# Print the total length of the text
print(f"Total length of the text: {len(text)}")

# Adjust the text slice to include more data
text = text[1000000:8000000]

# Print the length of the sliced text
print(f"Length of the sliced text: {len(text)}")

# Ensure the sliced text has sufficient length
if len(text) < 40000:  # Minimum length needed for a few sequences
    raise ValueError("The sliced text does not have enough data for sequence generation.")

# Create a sorted list of unique characters in the text
characters = sorted(set(text))

# Create dictionaries to map characters to indices and vice versa
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Set sequence length and step size
SEQ_LENGTH = 40
STEP_SIZE = 3

# # Prepare the input and output sequences
# sentences = []
# next_chars = []
# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i + SEQ_LENGTH])
#     next_chars.append(text[i + SEQ_LENGTH])

# # Debugging: Print the number of sequences
# print(f'Number of sequences: {len(sentences)}')

# # Initialize the input and output arrays
# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
# y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

# # Populate the input and output arrays
# for i, sentence in enumerate(sentences):
#     for t, character in enumerate(sentence):
#         x[i, t, char_to_index[character]] = 1
#     y[i, char_to_index[next_chars[i]]] = 1

# # Debugging: Print the shape of x and y
# print(f'Shape of x: {x.shape}')
# print(f'Shape of y: {y.shape}')

# # Check for non-zero entries in x and y
# print(f'Non-zero entries in x: {np.count_nonzero(x)}')
# print(f'Non-zero entries in y: {np.count_nonzero(y)}')

# # Build the model
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# # Train the model
# try:
#     model.fit(x, y, batch_size=256, epochs=4)
# except Exception as e:
#     print(f"An error occurred during training: {e}")

# # Save the model with a valid extension
# model.save('textgenerator.keras')


model = tf.keras.models.load_model('textgenerator.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index+SEQ_LENGTH]
    generated +=sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0,t,char_to_index[character]]=1
        
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        
        generated += next_character
        sentence = sentence[1:]+next_character
    return generated

print('---------------0.2--------------')
print(generate_text(300,0.2))
print('---------------0.4--------------')
print(generate_text(300,0.4))
print('---------------0.6--------------')
print(generate_text(300,0.6))
print('---------------0.8--------------')
print(generate_text(300,0.8))
print('---------------1--------------')
print(generate_text(300,1.0))


        
        