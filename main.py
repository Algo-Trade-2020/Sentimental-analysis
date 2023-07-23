from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import re 
from keras.utils import pad_sequences
import numpy as np 
model = load_model('sentimental.h5')

def clean_text(text):
    # Remove all characters except letters and numbers
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

texts = ['']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(len(sequence) for sequence in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

new_shape = (len(padded_sequences), 52)
new_arr = np.zeros(new_shape) 
new_arr[:, :len(padded_sequences[0])] = padded_sequences

print(np.argmax(model.predict(new_arr)))
print(np.argmin(model.predict(new_arr)))