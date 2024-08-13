import torch
import transformers
from transformers import AutoTokenizer, pipeline
from src.mistral_base_12 import evaluate_test_data_mistral7B
from src.utils.constants import TOKEN


def load_mistral_model(mistral_model_name, quantization_config):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        mistral_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        token=TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=TOKEN)
    return model, tokenizer


def evaluate_mistral_model(loaded_data, train_data, mistral_pipe, output_file, mistral_tokenizer):
    evaluate_test_data_mistral7B(loaded_data, train_data, mistral_pipe, output_file, mistral_tokenizer)


def create_pipeline(model, tokenizer):
    return pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=10)
