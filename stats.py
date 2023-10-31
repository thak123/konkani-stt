from datasets import load_dataset, DatasetDict, Dataset, Audio
import pandas as pd

import librosa




lang = "marathi"
model_name  = "openai/whisper-small"

# df = pd.read_csv("KonkaniCorpusDatasetRestructured.csv", sep="\t")
# df = pd.read_csv("KonkaniCorpusDatasetRestructuredRepeatingRemoved.csv", sep="\t")
df = pd.read_csv("KonkaniCorpusDatasetRestructuredNonRepeating.csv", sep="\t")

audio_dataset = Dataset.from_dict({"audio":df["audioFilename"].values.tolist()})

common_voice = audio_dataset.train_test_split(test_size=0.15,seed= 42)

total_duration = 0
# for i in common_voice["train"]["audio"]:
# 	total_duration+=librosa.get_duration(path=i)


import wave
for i in common_voice["train"]["audio"]:
	with wave.open(i) as mywav:
		duration_seconds = mywav.getnframes() / mywav.getframerate()
		total_duration+=duration_seconds
		print(f"Length of the WAV file: {duration_seconds:.1f} s")

print("total_duration",total_duration)