from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle

def save_tokenizer(tokenizer, filename='tokenizer_LJ.pickle'):
    with open(filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def clean_phoneme_tokens(texts):
    cleaned_texts = []
    for line in texts:
        tokens = line.split()
        cleaned_tokens = [token.strip("'") for token in tokens]  # Remove leading/trailing quotes
        cleaned_line = ' '.join(cleaned_tokens)
        cleaned_texts.append(cleaned_line)
    return cleaned_texts

# Load dataset
df = pd.read_csv('tts_data_LJ.csv')
texts = df['Phoneme_text'].values

# Clean phoneme tokens to unify things like 'hh and hh
cleaned_texts = clean_phoneme_tokens(texts)

# Initialize and train the tokenizer
tokenizer = Tokenizer(char_level=False)  # Word-level tokenizer for phonemes
tokenizer.fit_on_texts(cleaned_texts)

# Save the tokenizer
save_tokenizer(tokenizer)

# # Check the word index (phoneme to integer mapping)
print("Word Index:", tokenizer.word_index)

# # Convert text to sequences of integers
# sequences = tokenizer.texts_to_sequences(texts)
# # print("Sequences:", sequences)
# # Pad sequences to a fixed length (e.g., 10 for consistency in your model)
# max_sequence_length = 600
# padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
# print("Padded Sequences:", padded_sequences)
