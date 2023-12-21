import os
import time

import evaluate
import torch
import transformers
from transformers import AutoTokenizer, pipeline

from confidence_calculator import calculate_confidence
from handle_dataset import save_to_json, load_from_json
from utils.constants import results_test_trocr, results_LLM_mistral

cer_metric = evaluate.load('cer')


def correct_sentences(sentences, pipe):
    start = time.time()
    corrected_sentences = []
    # system_prompt = (
    #     "You are an expert on Washington's 18th century data, for example when you see a sentence like this: During "
    #     "here, no provision is to be de-, you correct it like this: During his stay here, no provision is to be de-.")
    for sentence in sentences:
        # user_prompt = (f"[INST] Correct the sentence, only change the wrong words that do not make sense and don't "
        #                f"change the right words to the sentence (if it is not necessary to correct the sentence, "
        #                f"you return it unchanged), without complete, without adding extra characters or information "
        #                f"and without adding new content: {sentence} [/INST]")
        # prompt = f"{system_prompt}\n {user_prompt}, then the correct sentence is:"

        prompt = f"Q: Is this valid English text? (y/n): ```{sentence}``` A: _|_"

        try:
            # nummer_length = 110
            nummer_length = 2*len(sentence) + 50
            corrected_text = pipe(prompt, max_length=nummer_length, do_sample=True, top_k=10, num_return_sequences=1,
                                  pad_token_id=pipe.tokenizer.eos_token_id)

            corrected_sentence = corrected_text[0]['generated_text'].split('then the correct sentence is:')[-1].strip()
            print(f'===== {corrected_sentence} ====')
            corrected_sentences.append(corrected_sentence if corrected_sentence else sentence)
        except Exception as e:
            print(f"Error in processing sentence '{sentence}': {e}")
            corrected_sentences.append(sentence)
    # Measure and print the time taken
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return corrected_sentences


def evaluate_test_data(loaded_data, pipe):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        actual_label = item['ground_truth_label']

        # Correct OCR predictions with Mistral
        mistral_text = correct_sentences([ocr_text], pipe)[0]

        # Compute CER for Mistral prediction
        cer_mistral = cer_metric.compute(predictions=[mistral_text], references=[actual_label])

        # Compute Confidence for Mistral prediction
    #     confidence_mistral = calculate_confidence(actual_label, mistral_text)
    #     results.append({
    #         'file_name': item['file_name'],
    #         'actual_label': actual_label,
    #         'OCR': {
    #             'predicted_label': ocr_text,
    #             'cer': item['cer'],
    #             'confidence': item['confidence']
    #         },
    #         'MISTRAL': {
    #             'predicted_label': mistral_text,
    #             'cer': cer_mistral,
    #             'confidence': confidence_mistral
    #         }
    #     })
    #
    # save_mistral_output = os.path.join(results_LLM_mistral, 'evaluation_results_with_mistral_v2_0.json')
    # save_to_json(results, save_mistral_output)


# Load Mistral model and tokenizer
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_model = transformers.AutoModelForCausalLM.from_pretrained(mistral_model_name)
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, device=torch.device('mps'))

results_path_from_ocr = os.path.join(results_test_trocr, 'test_evaluation_results_mini.json')
loaded_data = load_from_json(results_path_from_ocr)
# Example usage
evaluate_test_data(loaded_data, mistral_pipe)
print("The MISTRAL data is saved.")