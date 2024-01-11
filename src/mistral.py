import os
import time

import evaluate
import torch
import transformers
from transformers import AutoTokenizer, pipeline

from confidence_calculator import calculate_confidence
from handle_dataset import save_to_json, load_from_json
from utils.constants import results_test_trocr, results_LLM_mistral, results_LLM_mistral_1, WASHINGTON_SAMPLE_TEXT
device = torch.device("cpu")
cer_metric = evaluate.load('cer')


def calculate_pipe(pipe, prompt, nummer_length, top_k):
    return pipe(prompt, max_length=nummer_length, do_sample=True, top_k=top_k, num_return_sequences=1, pad_token_id=pipe.tokenizer.eos_token_id)


def check_grammar(sentence, pipe):
    prompt = f"Answer Yes or No: Is this sentence grammatically correct? '{sentence}', then the answer is:"

    try:
        nummer_length = 55

        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)

        response = corrected_text[0]['generated_text'].split(', then the answer is:')[-1].strip()
        if 'Yes' in response:
            grammar_check_results = 'Yes'
        elif 'No' in response:
            grammar_check_results = 'No'
        else:
            grammar_check_results = 'Error'
    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        grammar_check_results = 'Error'

    return grammar_check_results


def check_sentence(sentence, pipe):
    system_prompt = (
            f"Your ability as an AI includes fact checking and adjusting the language of historical texts to mirror "
            f"their respective epochs more closely, particularly texts from the George Washington dataset of the "
            f"18th century. Here is a sample: '{WASHINGTON_SAMPLE_TEXT}'.")
    try:
        user_prompt = (
            f"[INST] Authenticate and adjust this sentence to accurately reflect 18th-century language: '{sentence}' "
            f"[/INST]")
        prompt = f"{system_prompt}\n {user_prompt}\nThe revised sentence reads as:"
        nummer_length = (len(sentence)*2) + 480

        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)

        response = corrected_text[0]['generated_text'].split('The revised sentence reads as:')[-1].strip()
        # Post-processing to remove any additional unwanted text
        response = response.split('\n')[0].strip()

    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = 'Error'
    return response


def correct_sentences(sentences, pipe, batch_size=10):
    start = time.time()
    corrected_sentences = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_results = []
        for sentence in batch:
            try:
                check_grammar_ = check_grammar(sentence, pipe)
                if 'Yes' in check_grammar_:
                    batch_results.append(sentence)
                elif 'No' in check_grammar_:
                    batch_results.append(check_sentence(sentence, pipe))
                else:
                    batch_results.append('Error')
            except Exception as e:
                print(f"Error in processing sentence '{sentence}': {e}")
                batch_results.append('Error')
                corrected_sentences.append('Error')
        corrected_sentences.extend(batch_results)
    # Measure and print the time taken
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return corrected_sentences


def evaluate_test_data(loaded_data, pipe):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        confidence = item['confidence']

        # Correct OCR predictions with Mistral
        mistral_text = correct_sentences([ocr_text], pipe)[0]

        # Compute CER for Mistral prediction
        cer_mistral = cer_metric.compute(predictions=[mistral_text], references=[ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence': confidence
            },
            'MISTRAL': {
                'predicted_label': mistral_text,
                'cer': cer_mistral,
                'confidence': calculate_confidence(ground_truth_label, mistral_text)
            }
        })

    save_mistral_output = os.path.join(results_LLM_mistral_1, 'evaluation_results_with_mistral_10.json')
    save_to_json(results, save_mistral_output)


# Load Mistral model and tokenizer
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_model = transformers.AutoModelForCausalLM.from_pretrained(mistral_model_name, torch_dtype=torch.float16,
                                                                  device_map="auto")
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, batch_size=10)

results_path_from_ocr = os.path.join(results_test_trocr, 'test_evaluation_results.json')
loaded_data = load_from_json(results_path_from_ocr)
# Example usage
evaluate_test_data(loaded_data, mistral_pipe)
print("The MISTRAL data is saved.")
