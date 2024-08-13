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
from utils.constants import (results_test_trocr, results_mixed_LLM_MISTRAL, automated_resuts, TOKEN, EXAMPLES, \
    EXAMPLES_F, EXAMPLES_F_60, OCR_ERRORS, EXAMPLES_F_3, EXAMPLES_F_4, EXAMPLES_V2_F_10, EXAMPLES_O_V2_10, \
    EXAMPLES_MISRECOGNIZED_ERRORS, EXAMPLES_CORRECTOR_MISRECOGNIZED_ERRORS, EXAMPLES_INCORRECT_ABBREVIATIONS,
                             EXAMPLES_CORRECTOR_INCORRECT_ABBREVIATIONS)

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


def check_misrecognized_characters(text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR error identification. "
        f"Your task is to respond with 'Yes' or 'No' to the question: Does the given text line, which comes from "
        f"18th-century documents, contain misrecognized characters or number errors? "
        f"\n\n## Illustrative examples:"
        f"{EXAMPLES_MISRECOGNIZED_ERRORS}"
        f"\n[INST] Now, please analyze the following text line."
        f"Does the text line contain misrecognized characters or number errors? "
        f"The text line is: '{text_line}' [/INST]\nThen the output is:"
    )

    start = time.time()
    try:
        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 20

        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then the output is:"

        if json_output_marker in raw_response:
            json_output = raw_response.split(json_output_marker)[-1].strip()
            response_match = re.search(r"Response:\s*(Yes|No)", json_output)
            # explanation_match = re.search(r"Explanation:\s*(.*)", json_output)

            response = response_match.group(1).strip() if response_match else ""
            # explanation = explanation_match.group(1).strip() if explanation_match else ""

            if response == '':
                response = 'No'
            result = {
                "response": response,
                # "explanation": explanation
            }
        else:
            result = {"error": "JSON output marker not found in the response"}
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        result = {"error": str(e)}
    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, given the text line: {text_line} | "
                f"Misrecognized OCR errors: {result}")

    return result


def correct_misrecognized_characters(text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR correction. "
        f"Your task is to correct misrecognized characters or numbers in 18th-century documents"
        f"\n\n## Guidelines:"
        f"\n1. Address only identified misrecognized characters or numbers errors."
        f"\n2. Ensure corrections accurately reflect the 18th-century language and conventions."
        f"\n\n## Illustrative example:"
        f"{EXAMPLES_CORRECTOR_MISRECOGNIZED_ERRORS}"
        f"\n[INST] Given the text line: {text_line} [/INST]\nThen corrected text line is:"
    )
    start = time.time()
    try:
        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 25

        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then corrected text line is:"

        if json_output_marker in raw_response:
            result = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
        else:
            result = text_line
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        result = {"error": str(e)}
    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, given the text line: {text_line} | "
                f"Corrected text line: {result}")

    return result


def check_misinterpreted_words(text_line, pipe, mistral_tokenizer):
    # system_prompt = (
    #     f"Act as an 18th-century document analyst specializing in OCR error identification. "
    #     f"Your task is to respond with 'Yes' or 'No' to the question: Does the given text line, which comes from "
    #     f"18th-century documents, contain misinterpreted words? "
    #     f"\n\n## Illustrative examples:"
    #     f"{EXAMPLES_MISINTERPRETED_WORDS}"
    #     f"\n[INST] Now, please analyze the following text line."
    #     f"Does the text line contain misinterpreted words? "
    #     f"The text line is: '{text_line}' [/INST]\nThen the output is:"
    # )
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR error identification. "
        f"Your task is to respond with 'Yes' or 'No' to the question: Does the given text line, which comes from "
        f"18th-century documents, contain misrecognized characters or number errors?-- In addition, each text line "
        f"has a low CER value which means that most of the words are without misrecognised errors."
        f"\n\n## Illustrative examples:"
        f"{EXAMPLES_MISRECOGNIZED_ERRORS}"
        f"\n[INST] Now, please analyze the following text line."
        f"Does the text line contain misrecognized characters or number errors? "
        f"The text line is: '{text_line}' [/INST]\nThen the output is:"
    )

    start = time.time()
    try:
        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 20

        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then the output is:"

        if json_output_marker in raw_response:
            json_output = raw_response.split(json_output_marker)[-1].strip()
            response_match = re.search(r"Response:\s*(Yes|No)", json_output)

            response = response_match.group(1).strip() if response_match else ""

            if response == '':
                response = 'No'
            result = {
                "response": response,
            }
        else:
            result = {"error": "JSON output marker not found in the response"}
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        result = {"error": str(e)}
    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, given the text line: {text_line} | "
                f"Misinterpreted OCR errors: {result}")

    return result


def check_incorrect_abbreviations(text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR error identification. "
        f"Your task is to respond with 'Yes' or 'No' to the question: Does the given text line, which comes from "
        f"18th-century documents, contain incorrect abbreviations? "
        f"\n\n## Illustrative examples:"
        f"{EXAMPLES_INCORRECT_ABBREVIATIONS}"
        f"\n[INST] Now, please analyze the following text line."
        f"Does the text line contain incorrect abbreviations? "
        f"The text line is: '{text_line}' [/INST]\nThen the output is:"
    )

    start = time.time()
    try:
        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 20

        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then the output is:"

        if json_output_marker in raw_response:
            json_output = raw_response.split(json_output_marker)[-1].strip()
            response_match = re.search(r"Response:\s*(Yes|No)", json_output)

            response = response_match.group(1).strip() if response_match else ""

            if response == '':
                response = 'No'
            result = {
                "response": response,
            }
        else:
            result = {"error": "JSON output marker not found in the response"}
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        result = {"error": str(e)}
    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, given the text line: {text_line} | "
                f"Incorrect Abbreviations: {result}")

    return result


def correct_incorrect_abbreviations(text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR correction. "
        f"Your task is to correct incorrect abbreviations in 18th-century documents."
        f"\n\n## Guidelines:"
        f"\n1. Address only identified incorrect abbreviations errors."
        f"\n2. Ensure corrections accurately reflect the 18th-century language and conventions."
        f"\n\n## Illustrative example:"
        f"{EXAMPLES_CORRECTOR_INCORRECT_ABBREVIATIONS}"
        f"\n[INST] Given the text line: {text_line} [/INST]\nThen corrected text line is:"
    )
    start = time.time()
    try:
        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 25

        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then corrected text line is:"

        if json_output_marker in raw_response:
            result = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
        else:
            result = text_line
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        result = {"error": str(e)}
    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, given the text line: {text_line} | "
                f"Corrected text line: {result}")

    return result


def check_errors(text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR error identification. "
        f"Your task is to analyze the given text line and identify if it contains misrecognized characters, "
        f"incorrect abbreviations, or both. You need to parse the information and return one of the following "
        f"responses: "
        f"No error, misrecognized_characters, incorrect_abbreviations, or both."
        f"\n\n## Guidelines:"
        f"\n1. Identify misrecognized characters if characters are incorrect or replaced."
        f"\n2. Identify incorrect abbreviations if abbreviations do not adhere to 18th-century conventions."
        f"\n3. Ensure identifications accurately reflect the 18th-century language and conventions."
        f"\n\n## Illustrative examples:"
        f"\n1. Given the text line: 'Dear Sir, Oxon, Sept. 17, 1734.'"
        f"\nThe output is:"
        f"   Response: No error"

        f"\n2. Given the text line: 'D. James and Mr. S.'"
        f"\nThe output is:"
        f"   Response: incorrect_abbreviations"

        f"\n3. Given the text line: 'IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought'"
        f"\nThe output is:"
        f"   Response: misrecognized_characters"

        f"\n4. Given the text line: 'D. Jmes and Mr. Smth.'"
        f"\nThe output is:"
        f"   Response: both"

        f"\n[INST] Now, please analyze the following text line and provide your response."
        f"The text line is: '{text_line}' [/INST]"
        f"\nThen the output is:"
    )

    start = time.time()
    try:
        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 20

        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']
        json_output_marker = "Then the output is:"

        if json_output_marker in raw_response:
            json_output = raw_response.split(json_output_marker)[-1].strip()
            response_match = re.search(r"Response:\s*(No error|misrecognized_characters|incorrect_abbreviations|both)", json_output)

            response = response_match.group(1).strip() if response_match else ""

            result = {
                "response": response,
            }
        else:
            result = {"error": "JSON output marker not found in the response"}
    except Exception as e:
        logger.error(f"Error in processing text line '{text_line}': {e}")
        result = {"error": str(e)}
    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, given the text line: {text_line} | "
                f"Error Identification: {result}")

    return result


def check_and_correct_text_line(text_line, pipe, mistral_tokenizer):
    logger.debug(f"Checking and correcting text line: {text_line}")

    # Check for errors in the text line
    error_check_result = check_errors(text_line, pipe, mistral_tokenizer)
    error_type = error_check_result.get("response")

    # Initialize the corrected text as the original text
    corrected_text = text_line

    # Check for and correct misrecognized characters if identified
    if error_type in ["misrecognized_characters", "both"]:
        misrecognized_errors = check_misrecognized_characters(corrected_text, pipe, mistral_tokenizer)
        if misrecognized_errors.get("response") == "Yes":
            corrected_text = correct_misrecognized_characters(corrected_text, pipe, mistral_tokenizer)

    # Check for and correct incorrect abbreviations if identified
    if error_type in ["incorrect_abbreviations", "both"]:
        incorrect_abbreviations_errors = check_incorrect_abbreviations(corrected_text, pipe, mistral_tokenizer)
        if incorrect_abbreviations_errors.get("response") == "Yes":
            corrected_text = correct_incorrect_abbreviations(corrected_text, pipe, mistral_tokenizer)


    return corrected_text


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        corrected_text_line = check_and_correct_text_line(ocr_text, pipe, mistral_tokenizer)
        cer_mistral = evaluation.cer_only([corrected_text_line], [ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
            },
            'Prompt correcting': {
                'predicted_label': corrected_text_line,
                'cer': cer_mistral,
            }
        })

    save_mistral_output = os.path.join(automated_resuts, name_file)
    save_to_json(results, save_mistral_output)
    logger.info(f"Evaluation results saved at {save_mistral_output}")

# Load Mistral model and tokenizer
## mistral_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Hypothetical correct model name
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
