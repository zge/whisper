# Demo: transcription - long-form on English to English
#
# The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration.
# However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length.
# This is possible through Transformers pipeline method.
# Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline.
# It can also be extended to predict utterance level timestamps by passing return_timestamps=True
#
# Reference: https://huggingface.co/openai/whisper-large-v2
#
# Zhenhao Ge, 2023-02-02

import torch
from transformers import pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large-v2",
  chunk_length_s=30,
  device=device,
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# get prediction
sample = ds[0]["audio"]
prediction = pipe(sample)["text"]

# we can also return timestamps for the predictions
sample = ds[0]["audio"]
prediction = pipe(sample, return_timestamps=True)["chunks"]