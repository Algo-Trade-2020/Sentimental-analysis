import re
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

def clean_text(text):
    # Remove all characters except letters and numbers
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

# Assuming you have preprocessed text data and corresponding sentiment labels
data = pd.read_csv('all-data.csv', encoding='latin-1')
texts = data['News'].apply(clean_text) # Clean the text data
labels = data['class']  # List of sentiment labels (e.g., 'positive', 'negative', 'neutral')

# Rest of the code remains unchanged...
# Convert text to sequences and pad them to a fixed length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(len(sequence) for sequence in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert sentiment labels to numerical classes
label_to_num = {'positive': 0, 'negative': 1, 'neutral': 2}
num_labels = [label_to_num[label] for label in labels]
num_labels = np.array(num_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, num_labels, test_size=0.2, random_state=42)

# Build the LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_size = 100  # Size of the word embeddings
hidden_units = 128  # Number of LSTM units

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_sequence_length))
model.add(LSTM(hidden_units))
model.add(Dense( 64, 'relu'))
model.add(Dense( 32, 'relu'))
model.add(Dense( 16, 'tanh'))
model.add(Dense( 8, 'linear'))
model.add(Dense(3, activation='softmax'))  # 3 classes: positive, negative, neutral

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
model.save('sentimental.h5')
# Plot the accuracy and loss graphs
plt.figure(figsize=(12, 6))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
