import os
import json
import torch
import evaluate
from torch import optim
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    default_data_collator
from customOCRDataset import ModelConfig, CustomOCRDataset, DatasetConfig
from data_frame_handler import DataFrameHandler
from handle_dataset_washington import load_from_json, save_to_json
from src import evaluation
from utils.constants import outputs_path, model_save_path_seq_v2, processor_save_path_seq_v2
from utils.utils import clear_cuda_cache

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def calculate_sample_size(df, sample_percentage):
    handler = DataFrameHandler()
    training = load_from_json(df)
    train_df = handler.dict_to_dataframe(training)
    train_df = handler.sample_dataframe(train_df, sample_percentage).reset_index(drop=True)
    total_rows = len(train_df)
    return total_rows


def train_and_save_model(training_data_path, valid_data_path, model_save_dir, processor_save_dir, training_config,
                         sample_size=None):
    training = load_from_json(training_data_path)
    valid = load_from_json(valid_data_path)
    handler = DataFrameHandler()

    train_df = handler.dict_to_dataframe(training)
    valid_df = handler.dict_to_dataframe(valid)

    if sample_size:
        train_df = handler.sample_dataframe(train_df, sample_size).reset_index(drop=True)

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

    model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
    model.to(device)

    # Ensuring the correct configurations are set
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.bos_token_id = processor.tokenizer.cls_token_id

    # Set beam search parameters
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    def compute_cer(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = evaluation.cer_only([pred_str], [label_str])
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=training_config.BATCH_SIZE,
        per_device_eval_batch_size=training_config.BATCH_SIZE,
        output_dir=model_save_dir,
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=5,
        report_to='tensorboard',
        num_train_epochs=training_config.EPOCHS,
        fp16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator
    )

    res = trainer.train()
    print(res)

    # Save model, processor, and generation configuration
    os.makedirs(model_save_dir, exist_ok=True)
    model.save_pretrained(model_save_dir)
    processor.save_pretrained(processor_save_dir)
    print(f"Model saved to {model_save_dir}")
    print(f"Processor saved to {processor_save_dir}")

    clear_cuda_cache()