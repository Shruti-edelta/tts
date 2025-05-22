import pandas as pd
import numpy as np
import re
import unicodedata
import string
import contractions
import tensorflow as tf
import ast
from num2words import num2words


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag


class TextNormalizer:
    def __init__(self):
        self.abbreviations = {"Mr.": "Mister",
                        "Mrs.": "Misses",
                        "Dr.": "Doctor",
                        "No.": "Number",
                        "St.": "Street",
                        "Co.": "Company",
                        "Jr.": "Junior",
                        "Sr.": "Senior",
                        "Maj.": "Major",
                        "Gen.": "General",
                        "Drs.": "Doctors",
                        "Rev.": "Reverend",
                        "Lt.": "Lieutenant",
                        "Hon.": "Honorable",
                        "Sgt.": "Sergeant",
                        "Capt.": "Captain",
                        "Esq.": "Esquire",
                        "Ltd.": "Limited",
                        "Col.": "Colonel",
                        "Ft.": "Fort",
                        "Ave.": "Avenue",
                        "etc.": "et cetera",
                        "i.e.": "that is",
                        "e.g.": "for example",}
        # self.special_words = {
        #                 "https": "h t t p s",
        #                 "http": "h t t p",
        #                 "gmail": "g-mail",
        #                 "yahoo": "yahoo"}

    def expand_abbreviations(self, text):
        """Expands known abbreviations in the text."""
        # for abbr, expansion in self.abbreviations.items():
        #     # text = re.sub(r'\b'+abbr, expansion+" ", text)
        #     text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)  LJ002-0055

        for abbr, expansion in self.abbreviations.items():
            pattern = re.compile(re.escape(abbr), flags=re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    # def convert_numbers_to_words(self, text):
    #     """Convert numeric digits into words."""
    #     text = re.sub(r'\d+',w2n.word_to_num(text), text)
    #     return text
    def number_to_words(self,text):
        def replace(match):
            num = int(match.group())
            return num2words(num)
        return re.sub(r'\b\d+\b', replace, text)

    def tokenize_with_punctuation(self, text):
        """Tokenizes text, treating commas as separate tokens."""
        # This regex keeps alphanumeric sequences and commas as separate tokens
        return re.findall(r'\w+|[,.!?;:"\'()]', text)

    def remove_punctuation(self, text):
        """Remove punctuation marks from the text.(!"#$%&'()*+,-./:;<=>?@[\]^_`{|})"""
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def remove_extra_spaces(self, text):
        """Remove multiple spaces and trim leading/trailing spaces."""
        text = ' '.join(text.split())
        return text

    def expand_contractions(self, text):
        """Expand contractions like 'I'm' to 'I am'."""
        text = contractions.fix(text)
        return text

    def normalize_unicode(self, text):
        """Normalize unicode characters to a consistent form."""
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text

    def remove_urls_and_emails(self, text):
        """Remove URLs and email addresses from the text."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        return text
        
    def expand_urls_and_emails(self, text):        
        def expand_url(url):
            def expand_component(text):
                # Replace dots, slashes, and digits
                text = text.replace('.', ' dot ').replace('/', ' slash ')
                text = re.sub(r'\d+', lambda m: ' '.join(num2words(int(d)) for d in m.group()), text)
                return text

            if url.startswith("https://"):
                prefix = "h t t p s colon slash slash "
                url = url[len("https://"):]
            elif url.startswith("http://"):
                prefix = "h t t p colon slash slash "
                url = url[len("http://"):]
            else:
                prefix = ""

            normalized_url = prefix + expand_component(url)
            # print(normalized_url)
            normalized_url = re.sub(r'\s+', ' ', normalized_url).strip()
            return normalized_url
        
        text = re.sub(r'http\S+', lambda m: expand_url(m.group()), text)

        text = re.sub(r'([\w\.-]+)@([\w\.-]+)\.([a-z]{2,})',
                    lambda m: ''.join(m.group(1)) + ' at the rate ' + ''.join(m.group(2)) + ' dot ' + m.group(3),
                    text)

        return text

    def expand_alphanumeric(self, text):
        def replacer(match):
            token = match.group()
            # Split letters and digits
            return ' '.join([char.upper() if char.isalpha() else num2words(int(char)) for char in token])
        
        # Match tokens that have both letters and numbers
        return re.sub(r'\b(?=\w*[a-zA-Z])(?=\w*\d)\w+\b', replacer, text)

    # def expand_urls_and_emails(self, text):
    #     # for word, spoken in self.special_words.items():
    #     #     text = re.sub(r'\b' + word + r'\b', spoken, text, flags=re.IGNORECASE)
    #     # print(text)
    #     text = re.sub(r'https?://([\w.-]+)\.([a-z]{2,})(/[^\s]*)?',
    #                 lambda m: ' '.join(list('https')) + ' colon slash slash ' +
    #                             ''.join(m.group(1)) + ' dot ' + m.group(2) +
    #                             (' slash ' + m.group(3).replace('/', ' slash ') if m.group(3) else ''),
    #                 text)
        
    #     text = re.sub(r'([\w\.-]+)@([\w\.-]+)\.([a-z]{2,})',
    #                 lambda m: ''.join(m.group(1)) + ' at the rate ' + ''.join(m.group(2)) + ' dot ' + m.group(3),
    #                 text)
    #     return text

    def normalize_time(self,text):
        # Match time in 12-hour format (e.g., "10 am", "3:00 pm")
        text = re.sub(r'(\d{1,2}):(\d{2})\s*(am|pm)', r'\1:\2 \3', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d{1,2})\s*(am|pm)', r'\1:00 \2', text, flags=re.IGNORECASE)  # Standardize to HH:MM AM/PM
        return text
    
    # hamndle panctution (for pause ,rythm )
    def clean_text_for_g2p(text):
        text = text.lower()
        text = re.sub(r"[,]", " PAUSE ", text)  # preserve pause info
        text = re.sub(r"[.?!]", " PAUSE ", text)  # optional: treat sentence end as pause
        text = re.sub(r"[^a-zA-Z0-9\sPAUSE]", "", text)  # remove other punctuation
        text = re.sub(r"\s+", " ", text).strip()  # clean extra spaces
        return text
    
    # def tag_pos(self,text):
    #     return pos_tag(text)

    def tag_pos(self, text):
        tokens = word_tokenize(text)
        return pos_tag(tokens)

    def normalize_text(self, text):
        # text = text.lower()  # Lowercase the text
        text = self.expand_contractions(text)  # Expand contractions
        text = self.expand_abbreviations(text)  # Expand abbreviations
        text = self.remove_extra_spaces(text)  # Remove extra spaces
        text = self.normalize_unicode(text)  # Normalize unicode
        pos_tags = self.tag_pos(text)

        # text = self.normalize_time(text)  # Normalize time

        text = self.expand_urls_and_emails(text)  # Expand URLs and emails
        text = self.expand_alphanumeric(text) 
        text = self.number_to_words(text)  # Convert numbers to words

        text = self.remove_punctuation(text)  # Remove punctuation
        # return text
        return {
            "normalized_text":text,
            "pos_tags":pos_tags,
        }
    

    # def normalize_text_for_g2p(self, text):
    #     text = text.lower()  # Lowercase the text
    #     text = self.expand_contractions(text)  # Expand contractions
    #     text = self.expand_abbreviations(text)  # Expand abbreviations
    #     text = self.remove_extra_spaces(text)  # Remove extra spaces
    #     text = self.normalize_unicode(text)  # Normalize unicode
    #     text = self.expand_urls_and_emails(text)  # Expand URLs and emails
    #     text = self.expand_alphanumeric(text)
    #     text = self.number_to_words(text)  # Convert numbers to words
    #     tokens = self.tokenize_with_punctuation(text) # Tokenize, keeping commas
    #     # cleaned_tokens = self.remove_other_punctuation(tokens) # Remove other unwanted punctuation
    #     return tokens # Return a list of tokens, including comma

    # def normalize_text_for_pos_tagging(self, text):
    #     text = text.lower()
    #     text = self.expand_contractions(text)
    #     text = self.expand_abbreviations(text)
    #     text = self.remove_extra_spaces(text)
    #     text = self.normalize_unicode(text)
    #     return self.tag_pos(text)
    
    # def text_to_phonemes(self, text):
    #     """Convert text to phonemes."""
    #     text = self.normalize_text(text)  # Normalize the text first
    #     phonemes = []
    #     # Use phonemizer for ARPAbet or CMU Dictionary for word-level phonemes
    #     words = text.split()
    #     for word in words:
    #         if word_phonemes:
    #             phonemes.append(word_phonemes)
    #         else:
    #             # If phonemizer doesn't return a result, check CMU Pronouncing Dictionary
    #             cmu_phonemes = pronouncing.phones_for_word(word)
    #             if cmu_phonemes:
    #                 phonemes.append(cmu_phonemes[0])  # Take the first pronunciation variant
    #             else:
    #                 phonemes.append(word)
    #     return phonemes


class G2PConverter:
    # def __init__(self, model_path=None, vocab_path="dataset/G2P_dataset/cmu_dict_no_stress.csv", max_len=33, load_model=True):
    # def __init__(self, model_path=None, vocab_path="dataset/G2P_dataset/cmu_dict_pun_stress.csv", max_len=33, load_model=True):
    def __init__(self, model_path=None, vocab_path="dataset/G2P_dataset/cmu_dict_with_stress.csv", max_len=33, load_model=True):
        self.max_len = max_len
        if load_model and model_path:
            self.model = tf.keras.models.load_model(model_path)
        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path):
        df = pd.read_csv(vocab_path)
        self.words = df["word"].tolist()
        # self.phonemes = self._phoneme_string_to_list(df["phonemes"].tolist())
        self.phonemes = df["phonemes"].apply(ast.literal_eval).tolist()

        # Build grapheme vocab
        graphemes = sorted(set(ch for w in self.words for ch in str(w)))
        self.char2idx = {c: i + 1 for i, c in enumerate(graphemes)}
        self.char2idx['<pad>'] = 0
        self.char2idx['<sos>'] = len(self.char2idx)
        self.char2idx['<eos>'] = len(self.char2idx)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        print(self.char2idx)

        # Build phoneme vocab
        phoneme_set = sorted(set(p for ph in self.phonemes for p in ph))
        self.phn2idx = {p: i + 1 for i, p in enumerate(phoneme_set)}
        self.phn2idx['<pad>'] = 0
        self.phn2idx['<sos>'] = len(self.phn2idx)
        self.phn2idx['<eos>'] = len(self.phn2idx)
        self.idx2phn = {i: p for p, i in self.phn2idx.items()}
        print(self.phn2idx)

    def _phoneme_string_to_list(self, phoneme_strs):
        return [p.split() for p in phoneme_strs]

    def _encode_sequences(self, data, token2idx, maxlen=None):
        encoded = []
        for seq in data:
            s = [token2idx.get(c, token2idx['<pad>']) for c in seq]
            encoded.append(s)
        max_len = maxlen or max(len(s) for s in encoded)
        padded = [s + [token2idx['<pad>']] * (max_len - len(s)) for s in encoded]
        # print(padded)
        return np.array(padded)

    def preprocess_input(self, text):
        text = text.lower()
        words = text.split()
        encoded_words = [self._encode_sequences([w], self.char2idx, maxlen=self.max_len) for w in words]
        return encoded_words

    def predict(self, text):
        # print(text)
        encoded_words = self.preprocess_input(text)
        # predicted_phonemes = [[self.phn2idx['<sos>']]]
        predicted_phonemes = []
        pre_phoneme=[]
        for w in encoded_words:
            preds = self.model.predict(w, verbose=0)        # (1, 33, 42)
            phoneme_token = np.argmax(preds, axis=-1)[0]      # (33,) /(1, 33)
            # phoneme_token = [int(id) for id in phoneme_token if id != self.phn2idx['<pad>'] and id != self.phn2idx['<eos>']]
            phoneme_token = [int(id) for id in phoneme_token if id != self.phn2idx['<pad>']]
            phoneme_seq = [self.idx2phn.get(i, "<unk>") for i in phoneme_token if i != self.phn2idx['<pad>']] 
            # pre_phoneme.append(phoneme_seq)
            predicted_phonemes.append(phoneme_seq)
        # print("====",pre_phoneme)
        # predicted_phonemes.append([self.phn2idx['<eos>']])
        # print("======",predicted_phonemes)
        flat_phonemes = [p for word in predicted_phonemes for p in word] 
        return predicted_phonemes
    
    def batch_predict(self, texts):
        # Step 1: Preprocess each sentence into list of words
        batch_encoded = [self.preprocess_input(t) for t in texts]  # List[List[np.array]]
        # Step 2: Flatten all words from all sentences for batch prediction
        flat_inputs = [item for sentence in batch_encoded for item in sentence]
        # Step 3: Run prediction on the flat input
        preds = self.model.predict(np.vstack(flat_inputs), verbose=0)
        # Step 4: Convert predictions back to phonemes
        flat_results = []

        for pred in preds:
            phoneme_token = np.argmax(pred, axis=-1)
            phoneme_token = [int(id) for id in phoneme_token if id != self.phn2idx['<pad>'] and id != self.phn2idx['<eos>']]
            # phoneme_token = [int(id) for id in phoneme_token if id != self.phn2idx['<pad>']]
            phns = [self.idx2phn.get(i, "<unk>") for i in phoneme_token if i != self.phn2idx['<pad>']]
            # flat_results.append(phoneme_token)
            flat_results.append(phns)
        # Step 5: Reconstruct sentence-level phoneme list
        # print(flat_results)
        results = []
        i = 0
        for sentence in batch_encoded:
            # print(sentence)
            num_words = len(sentence)
            sentence_phonemes = flat_results[i:i+num_words]  # List of lists
            # print(sentence_phonemes)  
            # sentence_flat =[self.phn2idx['<sos>']] + [ph for word in sentence_phonemes for ph in word] + [self.phn2idx['<eos>']] #  sos(40) + Flatten word-level phonemes + eos(41)
            # sentence_flat =[ph for word in sentence_phonemes for ph in word] #  Flatten word-level phonemes 
            results.append(sentence_phonemes)
            i += num_words
        return results

if __name__ == "__main__":
    normalizer = TextNormalizer()
    raw_text = "Dr. Smith is going to the store. Visit http://www.example.com/test123//shruti.com/page or email me at test@example.com! I'm excited, No. 1 fan! c34545"
    # raw_text="Dr. Smith earned $5.6M in 2023."
    # raw_text="1. The male debtors' side consisted of a yard forty-nine feet by thirty-one,"
    # raw_text="5. The master felons' side. 6. The female felons' side. 7. The state side."
    # raw_text="A high wall fifteen feet in height divided the females' court-yard from the men's"
    # raw_text="This arrangement was, however, modified after eighteen eleven, and the chapel yard was allotted to misdemeanants and prisoners awaiting trial."
    # raw_text='long narrow rooms -- one thirty-six feet, six twenty-three feet, and the eighth eighteen i.e. e.g.'
    # raw_text="The numbers soon increased, however, and by 1811 had again risen to 629; and Mr. Neild was told that there had been at one time a4"
    # raw_text="He likewise indicated he was disenchanted with Russia."
    # raw_text="After summarizing the Bureau's investigative interest in Oswald prior to the assassination, J. Edgar Hoover concluded that, quote,"
    raw_text="Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
    # raw_text="directly under the chapel, in which there were three cells, used either for the confinement of disorderly and refractory prisoners"
    normalized_text = normalizer.normalize_text(raw_text)
    print("Original Text:", raw_text)
    print("Normalized Text:", normalized_text)

    g2p=G2PConverter("model/1/3model_cnn.keras")
    phonemes=g2p.predict(normalized_text['normalized_text'])
    print(phonemes)

    # texts = [
    #     "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
    #     "Hello world!",
    #     "The quick brown fox jumps over the lazy dog.",
    #     "G2P batch test.",
    #     "the recommendations we have here suggested would greatly advance the security of the office without any impairment of our fundamental liberties."
    # ]

#     normalized = [normalizer.normalize_text(t) for t in texts]
#     phonemes = g2p.batch_predict(normalized)
#     # print('========',phonemes)
#     for t, p in zip(texts, phonemes):
#         print(f"Text: {t}")
#         print(f"Phonemes: {p}")
#         print("---")

    # for word, phon in zip(normalized_text['normalized_text'].split(), phonemes):
    #     print(f"{word}: {' '.join(phon)}")


    # def ordinal_day(day):
    #     suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'st', 'nd', 'rd', 'th', 'th', 'th']
    #     if 4 <= day <= 20 or 24 <= day <= 30:
    #         return str(day) + suffix[0]
    #     else:
    #         return str(day) + suffix[day % 10]

    # # Function to normalize dates (convert to "5th March 2025" format)
    # def normalize_date(text):
    #     # Match dates in formats like "March 5th, 2025", "5th March 2025", "2025-03-05", etc.
    #     # We assume a date format like "5th March 2025" or "March 5, 2025"
    #     text = re.sub(r'(\d{1,2})\s([A-Za-z]+)\s(\d{4})', " dfd", text)
    #     return text





