import numpy as np
import os
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pandas as pd
# import noisereduce as nr # Uncomment if you're using noisereduce

# Configuration (adjust as needed)
# TARGET_SAMPLE_RATE = 22050 # Hz
# PROCESSED_AUDIO_DIR = 'my_tts_dataset/processed_audio'
# ... other paths and silence trimming parameters as defined previously ...

# Make sure the directory exists
# os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# Assuming you have a list of raw audio paths and their transcriptions
# For each raw_audio_path in your dataset:
#     1. Load audio with pydub (for easy format handling and basic ops)
#        audio = AudioSegment.from_file(raw_audio_path)

#     2. (Optional) Noise Reduction (using noisereduce, example only)
#        # Convert pydub AudioSegment to numpy array
#        audio_np = np.array(audio.get_array_of_samples())
#        if audio.sample_width == 2: # 16-bit
#            audio_np = audio_np.astype(np.float32) / (2**15)
#        elif audio.sample_width == 4: # 32-bit
#            audio_np = audio_np.astype(np.float32) / (2**31)
#        reduced_noise = nr.reduce_noise(y=audio_np, sr=audio.frame_rate)
#        # Convert back to pydub AudioSegment (careful with sample width)
#        audio = AudioSegment(
#            (reduced_noise * (2**15)).astype(np.int16).tobytes(),
#            frame_rate=audio.frame_rate,
#            sample_width=2,
#            channels=audio.channels
#        )


#     3. Convert to Mono
#        if audio.channels > 1:
#            audio = audio.set_channels(1)

#     4. Normalize Volume
#        audio = audio.normalize(-1.0) # Peak normalization to -1 dBFS

#     5. Silence Trimming
#        chunks = split_on_silence(audio, min_silence_len=..., silence_thresh=..., keep_silence=...)
#        if not chunks: continue # Skip if no speech detected
#        processed_audio_segment = AudioSegment.empty()
#        for chunk in chunks:
#            processed_audio_segment += chunk

#     6. Save as WAV at Target Sample Rate (using temp file and librosa for quality resampling)
#        temp_path = os.path.join(PROCESSED_AUDIO_DIR, "temp_" + os.path.basename(raw_audio_path))
#        processed_audio_segment.export(temp_path, format="wav")
#        y, sr = librosa.load(temp_path, sr=TARGET_SAMPLE_RATE)
#        sf.write(os.path.join(PROCESSED_AUDIO_DIR, os.path.basename(raw_audio_path)), y, TARGET_SAMPLE_RATE)
#        os.remove(temp_path) # Clean up temp file

#     Update your metadata file with the path to the processed audio file.

mel= np.load("audio/LJ001-0001.npy")

print(mel,mel.shape)

