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
    EXAMPLES_F, EXAMPLES_F_60, OCR_ERRORS, EXAMPLES_F_3, EXAMPLES_F_4

GRACE_TOKEN = 512
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


def check_sentence(sentence, pipe, mistral_tokenizer):
    logger.debug(f"Checking sentence: {sentence}")
    system_prompt = (
        "<s>"
        "[INST] Your primary task is to meticulously correct OCR (Optical Character Recognition) errors in a "
        "collection of 18th-century documents. These documents contain a variety of errors, ranging from simple "
        "misspellings to more complex issues like incorrect abbreviations and misinterpretations of terms. "
        "Your corrections must strive for precision, preserving the authenticity and integrity of the original "
        "manuscripts. It's imperative to avoid introducing new information or excluding essential details. "
        "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
        "linguistic conventions of the 18th century.[/INST]"
        "\n\n## Guidelines:"
        "\n- Address only OCR errors; please do not add more content."
        "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
        "\n- If you find cut-out words at the end of the OCR sentence, don't complete them."
        "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
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
        f"{EXAMPLES_F_4}"
        "[/INST]</s>"
    )

    adaptation_request = (
        f"<s>"
        f"[INST] Based on the guidelines, and the examples accurately correct all the OCR errors "
        f"in the following sentence: '{sentence}'.[/INST]</s>"
    )

    prompt = f"{system_prompt}\n{adaptation_request}\nThen the JSON output is:"
    tokens_prompt = count_tokens(prompt, mistral_tokenizer)
    nummer_length = tokens_prompt + GRACE_TOKEN
    logger.info(f"Number of total tokens: {nummer_length}")

    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        response = corrected_text[0]['generated_text'].split('Then the JSON output is:')[-1].strip()
        response = response.split('\n')[0].strip()
        # logger.info(f"Inside Try: response: {response}")
    except Exception as e:
        logger.error(f"Error in processing sentence '{sentence}': {e}")
        response = sentence
        # logger.info(f"Inside Exception: response: {response}")
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
            processed_sentence = check_sentence(sentence, pipe, mistral_tokenizer)
            if processed_sentence != 'Error':
                # Improved regex patterns to extract the necessary parts
                match_sentence = re.search(r"Corrected sentence:\s*(.*?)\s+Confidence", processed_sentence)
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


def check_label_correctness(sentence, pipe, mistral_tokenizer):
    prompt = (
        f"<s>[INST] You will be provided with a text line, and you need to respond with "
        f"'Yes' or 'No' to the question: Is the provided text line from the 18th century, that comes from an "
        f"OCR model, correct?\n"
        f"If you notice that it has OCR errors, then the text line is not correct.\n\n"
        f"Examples:\n"
        f"1. Sentence: 288. Letters Orders and Instructions Yuemler,\n"
        f"   Response: No\n"
        f"2. Sentence: This is a perfectly valid text line.\n"
        f"   Response: Yes\n"
        f"Given the text line: {sentence}\n\n"
        f"Please respond with 'Yes' or 'No' based on the correctness of the sentence.[/INST]</s>\n"
        f"Then the respond is:"
    )
    start = time.time()
    try:
        tokens_prompt = count_tokens(prompt, mistral_tokenizer)
        nummer_length = tokens_prompt + GRACE_TOKEN
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        if corrected_text and 'generated_text' in corrected_text[0]:
            response = corrected_text[0]['generated_text'].strip()
            response = response.split("Then the respond is:")[-1].strip()
            if 'Yes' in response:
                answer = 'Yes'
            elif 'No' in response:
                answer = 'No'
            else:
                answer = None

            if answer:
                result = {"response": answer}
            else:
                result = {"response": "Yes"}
        else:
            result = {"error": "Unexpected output format from calculate_pipe."}
    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        result = {"error": str(e)}
    end = time.time()
    print(f"Time taken: {end - start} seconds, Response: {result}")
    return result


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        is_correct = check_label_correctness(ocr_text, pipe, mistral_tokenizer)
        sentence_data = [{'sentence': ocr_text}]
        if "response" in is_correct and 'No' in is_correct["response"]:
            corrected_sentence, confidence, justification = correct_sentences(
                sentence_data,
                pipe,
                mistral_tokenizer
            )
            cer_mistral = evaluation.cer_only([corrected_sentence], [ground_truth_label])
        else:
            corrected_sentence = ocr_text
            confidence = "100"
            justification = "There is no need to correct it, because it is already correct."
            cer_mistral = item['cer']

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence': 0
            },
            'Prompt OCR checking': {
                'response': is_correct["response"],
            },
            'Prompt OCR correcting': {
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
