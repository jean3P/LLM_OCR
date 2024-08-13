import os
import time
import re
import evaluate
import torch
from outlines.models import transformers
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from handle_dataset_washington import save_to_json, load_from_json
from src import evaluation
from src.utils.logger import setup_logger
from utils.constants import results_test_trocr, results_mixed_LLM_MISTRAL, automated_resuts, TOKEN, EXAMPLES, \
    EXAMPLES_F, EXAMPLES_F_60, OCR_ERRORS, EXAMPLES_F_3, EXAMPLES_F_10, EXAMPLES_F_25, EXAMPLES_O_10, EXAMPLES_O_25

GRACE_TOKEN = 100
MODEL = "TrOCR"

device = torch.device("cuda")
cer_metric = evaluate.load('cer')

# Global variable to store context sentences for each document
document_contexts = {}

# Initialize logger
logger = setup_logger('workflow_logger', 'workflow.log')


def calculate_pipe(pipe, prompt, nummer_length, top_k):
    return pipe(prompt, max_length=nummer_length, do_sample=True, top_k=top_k, num_return_sequences=1,
                pad_token_id=pipe.tokenizer.eos_token_id, truncation=True)


def count_tokens(prompt, mistral_tokenizer):
    input_ids = mistral_tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.shape[1]


def check_text_line(text_line, pipe, mistral_tokenizer):
    logger.debug(f"Checking text line: {text_line}")
    system_prompt = (
        "<s>"
        "[INST] Act as an 18th-century document analyst specializing in OCR correction with meticulous attention to "
        "historical accuracy and authenticity. "
        "Your primary task is to perform OCR error correction for 18th-century documents. "
        "Ensure corrections maintain the original style, historical accuracy, and adhere to linguistic conventions of "
        "the 18th century. "
        "Remember, do not introduce new information or exclude essential details. "
        "Address only OCR errors, ensure the corrected text line is not excessively longer than the original, and "
        "remove repeated words. "
        "Incorporate contextual cues and examples provided.[/INST]"
        "\n\n## Guidelines:"
        "\n- Address only OCR errors; please do not add more content."
        "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
        "\n- If you find cut-out words at the end of the OCR text line, don't complete them."
        "\n- If the corrected text line is twice the length of the OCR text line, return the OCR text line. "
        "\n- The confidence percentage reflects your certainty in the accuracy and "
        "error-free nature of the corrected text. "
        "\n- Remove repeated words."
        "\n- Correct misrecognized characters or numbers."
        "\n- Correct misinterpreted words based on context."
        "\n- Adjust incorrect abbreviations or terms to their correct historical usage."
        "\n- Correct misplaced or extra characters."
        "\n- Add or correct punctuation as needed."
        "\n- Standardize formatting to match historical conventions."
        "\n\n## Illustrative examples:"
        f"{EXAMPLES_O_25}"
        "[/INST]</s>"
    )

    adaptation_request = (
        f"<s>"
        f"[INST] Based on the guidelines, and the examples accurately correct all the OCR errors "
        f"in the following text line: '{text_line}'.[/INST]</s>"
    )

    prompt = f"{system_prompt}\n{adaptation_request}\nThen the JSON output is:"
    tokens_prompt = count_tokens(prompt, mistral_tokenizer)
    nummer_length = tokens_prompt + GRACE_TOKEN
    logger.info(f"Number of total tokens: {nummer_length}")

    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        response = corrected_text[0]['generated_text'].split('Then the JSON output is:')[-1].strip()
        response = response.split('\n')[0].strip()
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        response = text_line
    return response


def correct_sentences(sentence_data, pipe, mistral_tokenizer, batch_size=10):
    start = time.time()
    corrected_sentence = ''
    confidence = ''
    justification = ''

    for i in range(0, len(sentence_data), batch_size):
        batch = sentence_data[i:i + batch_size]
        for data in batch:
            sentence = data['sentence']
            logger.debug(f"Processing batch sentence: {sentence}")
            processed_sentence = check_text_line(sentence, pipe, mistral_tokenizer)
            if processed_sentence != 'Error':
                # Improved regex patterns to extract the necessary parts
                match_sentence = re.search(r"Corrected text line:\s*(.*?)\s+Confidence", processed_sentence)
                match_confidence = re.search(r"Confidence \(%\):\s*(\d+)", processed_sentence)
                match_justification = re.search(r"Justification:\s*(.*)", processed_sentence)

                if match_sentence:
                    corrected_sentence = match_sentence.group(1).strip()
                else:
                    corrected_sentence = sentence  # Fallback text if no sentence is found

                if match_confidence:
                    confidence = match_confidence.group(1).strip()
                else:
                    confidence = '0'

                if match_justification:
                    justification = match_justification.group(1).strip()
                else:
                    justification = 'No justification provided, sentence is returned uncorrected.'
            else:
                corrected_sentence = sentence  # return original if error
                confidence = '0'
                justification = 'Sentence is returned uncorrected.'

            logger.info(
                f"OCR sentence: {sentence} | Corrected Sentence: {corrected_sentence} | "
                f"Confidence: {confidence} | Justification: {justification}")

    end = time.time()
    logger.info(f"Time taken for batch: {end - start} seconds")
    return corrected_sentence, confidence, justification


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        sentence_data = [{'sentence': ocr_text}]

        corrected_sentence, confidence, justification = correct_sentences(sentence_data,
                                                                          pipe,
                                                                          mistral_tokenizer,
                                                                          10)

        # cer_mistral = cer_metric.compute(predictions=[corrected_sentence], references=[ground_truth_label])

        cer_mistral = evaluation.cer_only([corrected_sentence], [ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence': 0
            },
            'Prompt correcting': {
                'predicted_label': corrected_sentence,
                'cer': cer_mistral,
                'confidence': int(confidence),
                'justification': justification
            }
        })

    save_mistral_output = os.path.join(automated_resuts, name_file)
    save_to_json(results, save_mistral_output)
    logger.info(f"Evaluation results saved at {save_mistral_output}")

# # Load Mistral model and tokenizer
# ## mistral_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Hypothetical correct model name
# mistral_model_name = "mistralai/Mistral-7B-v0.1"
# mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_name,
#                                                      torch_dtype=torch.float16,
#                                                      device_map="auto",
#                                                      token=TOKEN
#                                                      )
# mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=TOKEN, max_length=32)
# mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, batch_size=10)
#
# results_path_from_ocr = os.path.join(results_test_trocr, 'testing.json')
# loaded_data = load_from_json(results_path_from_ocr)
# ## Example usage
# evaluate_test_data_mistral7B(loaded_data, mistral_pipe, 'final_3.json', mistral_tokenizer)
# print("The MISTRAL data is saved.")
