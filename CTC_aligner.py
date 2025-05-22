import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

bundle = WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
labels = bundle.get_labels()

from ctc_segmentation import CtcSegmentationParameters, prepare_token_list, ctc_segmentation, determine_utterance_segments

params = CtcSegmentationParameters()
params.char_list = labels  # From bundle.get_labels()

# Merge phoneme sequence into string with blanks inserted for CTC
ground_truth, utt_begin_indices = prepare_token_list(params, [' '.join(phonemes)])

# Extract audio
waveform, _ = torchaudio.load("output.wav")

# Get emission matrix
with torch.inference_mode():
    emissions, _ = model(waveform)

emission = emissions[0].cpu().numpy()

# Run segmentation
timings, char_probs, state_list = ctc_segmentation(params, emission, ground_truth)

# Get actual segments (start/end times for each phoneme)
segments = determine_utterance_segments(utt_begin_indices, timings, char_probs, state_list)