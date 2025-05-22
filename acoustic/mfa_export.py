import os
import ast
from collections import defaultdict

def export_mfa_files(df, output_dir="dataset/LJSpeech"):
    """
    Generate .txt, .lab, and dict.txt for MFA from a DataFrame
    with columns: File_Name, Normalize_text, Phoneme_text (list of phoneme lists).
    """

    txt_dir = os.path.join(output_dir, "txt")
    lab_dir = os.path.join(output_dir, "lab")
    dict_path = os.path.join(output_dir, "dict.txt")

    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    word_to_phoneme = defaultdict(set)
    print(word_to_phoneme)
    for _, row in df.iterrows():
        utt_id = row['File_Name']
        text = row['Normalize_text'].strip()
        phoneme_seq = row['Phoneme_text']

        # Ensure phoneme_seq is a list of lists
        if isinstance(phoneme_seq, str):
            phoneme_seq = ast.literal_eval(phoneme_seq)

        # Save .txt
        with open(os.path.join(txt_dir, f"{utt_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # Save .lab (flatten and append <eos>)
        flat_phonemes = [ph for word in phoneme_seq for ph in word]
        flat_phonemes.append("<eos>")
        with open(os.path.join(lab_dir, f"{utt_id}.lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(flat_phonemes))

        # Populate dictionary
        for word, phonemes in zip(text.lower().split(), phoneme_seq):
            word_to_phoneme[word].add(" ".join(phonemes))

    # Write dict.txt
    with open(dict_path, "w", encoding="utf-8") as f:
        for word, variants in sorted(word_to_phoneme.items()):
            for variant in variants:
                f.write(f"{word} {variant}\n")

    print(f"✅ Exported .txt files to: {txt_dir}")
    print(f"✅ Exported .lab files to: {lab_dir}")
    print(f"✅ Exported dict.txt to: {dict_path}")




    '''mfa align \
  dataset/mfa_data/wavs_16k \
  dataset/mfa_data/dict.txt \
  english_mfa \
  output/LJSpeech_alignments \
  --lab_dir dataset/mfa_data/lab \
  --use_phone_input \
  --use_single_speaker \
  --clean \
  --debug

  mfa train dataset/mfa_data1 dataset/mfa_data1/dict.txt output/acoustic_model --no_lda --single_speaker --clean
  
  '''