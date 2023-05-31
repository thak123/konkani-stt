from datasets import Dataset, Audio
import pandas as pd

from os.path import exists

print("hi")
df = pd.read_csv("KonkaniCorpusDatasetRestructured.csv", sep="\t")
print(df.describe())
audio_dataset = Dataset.from_dict({"audio":df["audioFilename"].values.tolist(), 
                                   "sentence":df["sentences"].astype(str).values.tolist()}).cast_column("audio", Audio())

with open("tmp.txt","w") as inputfile:
    for i,j in zip(df["audioFilename"].values.tolist(), df["sentences"].values.tolist()):
        try:
            # ds = Dataset.from_dict({"audio":[i], "sentence":[j]}).cast_column("audio", Audio())
            if not exists(i):
                print(i)
        except:
            print(i)
            inputfile.write(i+"\n")
            inputfile.flush()
