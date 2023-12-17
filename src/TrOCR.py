import glob
import os
from dataclasses import dataclass

import evaluate
import torch
from torch import optim
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, default_data_collator, \
    Seq2SeqTrainer

from customOCRDataset import ModelConfig, CustomOCRDataset, DatasetConfig
from data_frame_handler import DataFrameHandler
from handle_dataset import load_from_json
from utils.constants import outputs_path, model_save_path, processor_save_path

device = torch.device('mps')

training = load_from_json(os.path.join(outputs_path, 'train', 'training_data.json'))
valid = load_from_json(os.path.join(outputs_path, 'valid', 'validation_data.json'))

handler = DataFrameHandler()

train_df = handler.dict_to_dataframe(training)
valid_df = handler.dict_to_dataframe(valid)

sampled_train_df = handler.sample_dataframe(train_df, 10)


@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 48
    EPOCHS: int = 9  # 35
    LEARNING_RATE: float = 0.00005


"""
Initializing the processor for data preprocessing and tokenization. Creating datasets for training and validation.
"""

processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
train_dataset = CustomOCRDataset(
    root_dir=os.path.join(DatasetConfig.DATA_ROOT),
    df=train_df,
    processor=processor
)
valid_dataset = CustomOCRDataset(
    root_dir=os.path.join(DatasetConfig.DATA_ROOT),
    df=valid_df,
    processor=processor
)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

"""
Loading a pre-trained VisionEncoderDecoder model specifically designed for OCR tasks. 
The model is configured for the specific use case.
"""

model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
model.to(device)
print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Set special tokens used for creating the decoder_input_ids from the labels.
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# Set Correct vocab size.
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id

model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

"""
Setting up the optimizer for model training and defining the Character 
    Error Rate (CER) metric for performance evaluation.
"""

optimizer = optim.AdamW(
    model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.0005
)

cer_metric = evaluate.load('cer')


def compute_cer(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
    per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
    output_dir='seq2seq_model_printed/',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    report_to='tensorboard',
    num_train_epochs=TrainingConfig.EPOCHS
)

# Initialize trainer.
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_cer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=default_data_collator
)

"""
Starting the training process of the model and printing the results after training is completed.
"""
res = trainer.train()

print(res)

# Create the directories if they don't exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(processor_save_path, exist_ok=True)

# Save the model
model.save_pretrained(model_save_path)

# Save the processor
processor.save_pretrained(processor_save_path)

print(f"Model saved to {model_save_path}")
print(f"Processor saved to {processor_save_path}")
