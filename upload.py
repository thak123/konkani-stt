from transformers import Seq2SeqTrainer,AutoModel

from transformers import Seq2SeqTrainingArguments
from huggingface_hub import login

login("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")

training_args = Seq2SeqTrainingArguments(
  output_dir ="whisper-small-gom"
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=AutoModel.from_pretrained("whisper-small-gom/checkpoint-4000"),
  
)

kwargs = {
    "dataset_tags": "newsonair_konkani_external_aligned_lab_02-09-2021_06-55",
    "dataset": "newsonair_konkani",  # a 'pretty' name for the training dataset
    "dataset_args": "config: mr, split: test",
    "language": "mr",
    "model_name": "Whisper Small Gom - Gaurish Thakka",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
    'hub_model_id':"gom-konkani/whisper-small-gom"
}

# from transformers import PushToHubCallback

# push_to_hub_callback = PushToHubCallback(
#     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
# )

trainer.push_to_hub(**kwargs)
