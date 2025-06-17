import pandas as pd
import ast

class CMUDictCSV:
    def __init__(self, csv_path):
        self.word_to_phonemes = {}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            word = str(row['word']).lower()
            phonemes = ast.literal_eval(row['phonemes'])
            self.word_to_phonemes[word] = phonemes

    def get(self, word):
        return self.word_to_phonemes.get(word.lower(), None)

class HybridG2P:
    def __init__(self, g2p_model, cmu_dict):
        self.g2p_model = g2p_model  # Your trained G2PConverter
        self.cmu_dict = cmu_dict    # CMUDictCSV instance

    def word_to_phonemes(self, word):
        # Try CMU dict
        dict_result = self.cmu_dict.get(word)
        if dict_result is not None:
            return dict_result

        # Fallback to model
        return self.g2p_model.predict(word)

    def text_to_phonemes(self, text):
        words = text.strip().split()
        phoneme_seq = []
        for word in words:
            phonemes = self.word_to_phonemes(word)
            phoneme_seq.extend(phonemes)
        # Optional: add <eos> or other special tokens
        phoneme_seq.append(self.g2p_model.phn2idx['<eos>'])
        return phoneme_seq


# from text_preprocess import G2PConverter  # Your model

# Load trained model
# g2p_model = G2PConverter(model_path="model/1/3model_cnn.keras")

# # Load dictionary
# import nltk
# nltk.download('cmudict')
# from nltk.corpus import cmudict
# cmu_dict = cmudict.dict()

# # Create Hybrid G2P
# hybrid = HybridG2P(g2p_model=g2p_model, cmu_dict=cmu_dict)

# # Convert text
# text = "shruti is reading"
# phoneme_sequences = hybrid.text_to_phonemes(text)
# print(phoneme_sequences)
