from datasets import load_dataset, DatasetDict, Dataset, Audio, load_from_disk
import pandas as pd
from transformers import EarlyStoppingCallback
# from datasets.utils.logging import disable_progress_bar
# disable_progress_bar()

SEED = 1042 #10 #42
from transformers import set_seed

set_seed(SEED)
print(SEED)

lang = "marathi"
model_name  = "openai/whisper-large-v2"

# df = pd.read_csv("KonkaniCorpusDatasetRestructured.csv", sep="\t")
# df = pd.read_csv("KonkaniCorpusDatasetRestructuredRepeatingRemoved.csv", sep="\t")
df = pd.read_csv("KonkaniCorpusDatasetRestructuredNonRepeating.csv", sep="\t")

audio_dataset = Dataset.from_dict({"audio":df["audioFilename"].values.tolist(), 
                                   "sentence":df["sentences"].astype(str).values.tolist()}).cast_column("audio", Audio(sampling_rate=16000))

train_testvalid = audio_dataset.train_test_split(test_size=0.15,seed= SEED)
test_valid = train_testvalid['test'].train_test_split(test_size=0.1,seed= SEED)

common_voice = DatasetDict({
    'train': train_testvalid['train'],
    'validation': test_valid['test'],
    'test': test_valid['train']
})

# common_voice = audio_dataset.train_test_split(test_size=0.15,seed= SEED)

print(common_voice)

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name, language=f"{lang}", task="transcribe")


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name, language=f"{lang}", task="transcribe")


print(common_voice["train"][0])

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])

max_label_length = 448

def filter_labels(labels):
    """Filter label sequences longer than max length"""
    return len(labels) < max_label_length

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# common_voice = common_voice.map(prepare_dataset, 
#                                 remove_columns=common_voice.column_names["train"],
#                                 num_proc=4)

# common_voice = common_voice.filter(filter_labels, input_columns=["labels"])

# common_voice.save_to_disk("whisper-dataset")
common_voice = load_from_disk("whisper-dataset")

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.dropout = 0.3 #0.1 #0.2
# to use gradient checkpointing
# model.config.max_length = 512
model.config.use_cache = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)
print(model)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{model_name}--{SEED}",  # gom-LDC-v1.non-repeating" change to a repo name of your choice
    per_gpu_train_batch_size=8,#16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    eval_accumulation_steps=1,
    
    learning_rate=0.8e-5,
    warmup_steps=500,#500,
    # max_steps=2000, #8000,#4000,
    # gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="steps",
    per_gpu_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=500,
    num_train_epochs=5,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", #"wer",
    # greater_is_better=True,#False,
    push_to_hub=False, #ToDo
    # optim="adamw_bnb_8bit",
   
)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

)

processor.save_pretrained(training_args.output_dir)

trainer.train()

trainer.evaluate(eval_dataset = common_voice["test"])
