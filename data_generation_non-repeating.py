#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os


# In[ ]:


df = pd.read_csv("KonkaniCorpusDataset.csv",sep="\t")


# In[ ]:


df


# In[ ]:


checklist = ["Command and Control Words-W1","Form and Function Word-W5",
             "Most Frequent Word-FullSet-W3B","Most Frequent Word-Part-W3A",
             "Person Name-W2","Phonetically Balanced-W4",
             "Place Name-W2"]
checklist


# In[ ]:





# In[49]:


new_sentences = []
audio_file_names = []

for index, row in df.iterrows():
    search_term = row['txt_path'].split("/")[3]
    if search_term in checklist: 
#         print(row['sentence'], row['txt_path'])
        # new_sentence = row['sentence'] + " " +row['sentence'] + " " +row['sentence']
        new_sentences.append(row['sentence'])
    else:
        new_sentences.append(row['sentence'])
    
    audio_file_name = row['txt_path'].split('.txt')[0]+".wav"
    print(audio_file_name)
    audio_file_names.append(audio_file_name)
   


# In[50]:


len(new_sentences), df.shape,len(audio_file_names)


# In[51]:


audio_file_names


# In[52]:


df["audioFilename"] =  audio_file_names


# In[53]:


df["new_sentence"] = new_sentences


# In[54]:


df.sample(4).head()


# In[55]:


df = df[["new_sentence","audioFilename"]]


# In[59]:


df = df.rename(columns={"new_sentence":"sentences"})
df


# In[60]:


df.to_csv("KonkaniCorpusDatasetRestructuredNonRepeating.csv",sep="\t",index=False)


# In[61]:


df

from datasets import Dataset, Audio
import pandas as pd

from os.path import exists

print("hi")
df = pd.read_csv("KonkaniCorpusDatasetRestructuredNonRepeating.csv", sep="\t")
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


# In[ ]:




