import json
import os
import time
import re
from collections import defaultdict
from difflib import get_close_matches

import evaluate
import torch
from outlines.models import transformers
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from handle_dataset_washington import save_to_json, load_from_json
from src import evaluation
from src.utils.logger import setup_logger
from utils.constants import (results_test_trocr, results_mixed_LLM_MISTRAL, automated_resuts, TOKEN, EXAMPLES, \
                             EXAMPLES_F, EXAMPLES_F_60, OCR_ERRORS, EXAMPLES_F_3, EXAMPLES_F_4, EXAMPLES_V2_F_10,
                             EXAMPLES_O_V2_10, \
                             EXAMPLES_MISRECOGNIZED_ERRORS, EXAMPLES_CORRECTOR_MISRECOGNIZED_ERRORS,
                             EXAMPLES_INCORRECT_ABBREVIATIONS,
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
                pad_token_id=pipe.tokenizer.eos_token_id)


def count_tokens(prompt, mistral_tokenizer):
    input_ids = mistral_tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.shape[1]


# Load Data from JSON Files
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Extract text lines from training data
def extract_text_lines_from_train_data(train_data):
    return list(train_data.values())


# Check if a word contains mostly digits
def is_mostly_numeric(word):
    digits_count = sum(char.isdigit() for char in word)
    return digits_count > len(word) / 2


# Function to find the top 3 most similar words, excluding words that are mostly numbers and removing duplicates
# Function to find the top 3 most similar words, excluding words that are mostly numbers and removing duplicates
def find_top_3_similar_words(word, train_set_lines, is_start_of_line=False):
    def split_suggestion(suggestion, original):
        if original.endswith('-') and not suggestion.endswith('-'):
            return suggestion[:len(original)-1] + '-'
        if is_start_of_line and not suggestion.startswith(original):
            return original
        return suggestion

    all_words = [line.split() for line in train_set_lines]
    flat_list = [item for sublist in all_words for item in sublist]
    matches = get_close_matches(word, flat_list, n=3, cutoff=0.85)
    unique_matches = list(dict.fromkeys(matches))  # Remove duplicates while preserving order
    split_matches = [split_suggestion(match, word) for match in unique_matches]
    # logger.info(f"Finding top 3 similar words for '{word}' -> {split_matches}")
    return split_matches if split_matches else [word]


# Detect OCR Errors and Suggest Corrections
def detect_errors_and_suggest_corrections(ocr_text, train_set_lines):
    suggestions = []
    words = ocr_text.split()
    for idx, word in enumerate(words):
        is_start_of_line = (idx == 0)
        similar_words = find_top_3_similar_words(word, train_set_lines, is_start_of_line)
        suggestions.append((word, similar_words))
    logger.info(f"OCR text '{ocr_text}' suggestions: {suggestions}")
    return suggestions


# Correct Text Line with Context
def correct_text_with_context(ocr_text, suggestions, pipe, mistral_tokenizer):

    # if all(len(similar_words) == 1 and similar_words[0] == ocr_word for ocr_word, similar_words in suggestions):
    #     return ocr_text

    suggestion_part = "\n".join(
        f"Original word: {ocr_word}, Suggestions: {', '.join(set(similar_words))}"
        for ocr_word, similar_words in suggestions if similar_words and set(similar_words) != {ocr_word}
    )

    if not suggestion_part:
        suggestion_part = "No suggestions available."

    # logger.info(f"Suggestions: {suggestion_part}")

    system_prompt = (
        f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
        f"Your task is to correct misrecognized characters or numbers in 18th-century documents."
        f"\n\n## Guidelines:"
        f"\n1. Ensure corrections accurately reflect the 18th-century language and conventions"
        f"\n2. You need to maintain the same length of the text line"
        f"\n3. Be sure to keep punctuation marks of the original text line and do not add new punctuation that is "
        f"not in the original line of text"
        f"\n4. Preserve original word splits or cuts (e.g., 'incomple-' should not be combined into 'incomplete')"
        f"\n5. If the original text line is hyphenated, you have to keep the hyphen "
        f"\n6. Don't delete words that are not duplicated"
        f"\n7. Words cut off at the end with a hyphen should not be completed"
        f"\n\n## Suggestions of similar words from the training set:"
        f"\n{suggestion_part}"
        f"\nBased on the Guidelines and suggestions, correct the text line: {ocr_text} [/INST]"
        f"\nThen corrected text line is:"
    )

    tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 25
    response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
    raw_response = response[0]['generated_text']
    json_output_marker = "Then corrected text line is:"

    if json_output_marker in raw_response:
        result = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
    else:
        result = ocr_text

    # Remove content within square brackets and extra spaces
    corrected_text = re.sub(r'\[.*?\]', '', result).strip()
    corrected_text = re.sub(r'\s+', ' ', corrected_text)

    # Ensure the length is within reasonable bounds
    if len(corrected_text) > (len(ocr_text) * 1.5):
        corrected_text = ocr_text
    elif len(corrected_text) < (len(ocr_text) / 2):
        corrected_text = ocr_text

    logger.info(f"Corrected text for '{ocr_text}': {corrected_text}")
    return corrected_text


def check_misrecognized_characters(text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"Act as an 18th-century document analyst specializing in OCR error identification. "
        f"Your task is to respond with 'Yes' or 'No' to the question: Does the given text line, which comes from "
        f"18th-century documents, contain misrecognized characters or number errors?-- In addition, each text line "
        f"has a low CER value which means that most of the words are without misrecognised errors. Words cut off at "
        f"the beginning of the line are not misrecognised errors."
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


def find_immediate_repeated_words(text_line):
    """
    This function takes a text line as input and returns a list of immediately repeated words found in the text line.
    This includes cases where there is punctuation or whitespace between the repeated words, and insensitive.
    """
    # Match words that are repeated with or without punctuation in between, insensitive
    repeated_words = re.findall(r'\b(\w+)\b[\s\W]+\b\1\b', text_line, flags=re.IGNORECASE)
    # # Ensure the found duplicates are indeed duplicates
    # actual_repeats = [match for match in repeated_words if text_line.lower().count(match.lower()) > 1]
    return repeated_words


def find_close_repeated_word_sets(text_line):
    """
    This function takes a text line as input and returns a list of sets of closely repeated words found in the text line.
    It detects repeated sequences of words in the text line.
    """
    words = re.findall(r'\b\w+\b', text_line)
    repeated_word_sets = []

    # Check for repeated sequences
    for length in range(2, len(words) // 2 + 1):  # Sequence lengths from 2 to half the length of the words
        sequence_positions = defaultdict(list)

        # Record positions of each word sequence
        for i in range(len(words) - length + 1):
            sequence = ' '.join(words[i:i + length]).lower()
            sequence_positions[sequence].append(i)

        # Identify closely repeated word sets
        for positions in sequence_positions.values():
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    if positions[i + 1] - positions[i] <= length:
                        repeated_set = ' '.join(words[positions[i]:positions[i] + length])
                        if repeated_set not in repeated_word_sets:
                            repeated_word_sets.append(repeated_set)

    return repeated_word_sets


def correct_duplicated_words_prompt(text_line, pipe, mistral_tokenizer):
    # First, use the find_immediate_repeated_words function to detect repeated words
    immediate_duplicated_words = find_immediate_repeated_words(text_line)
    close_repeated_word_sets = find_close_repeated_word_sets(text_line)

    if not immediate_duplicated_words and not close_repeated_word_sets:
        # If no duplicated words are found, return the original text line
        logger.info(f"No duplicated words found for text: {text_line}")
        return text_line

    # Combine both sets of duplicated words for the prompt
    duplicated_words_part = "\n".join(
        f"Duplicated word: {word}" for word in immediate_duplicated_words
    )
    repeated_sets_part = "\n".join(
        f"Repeated set: {word_set}" for word_set in close_repeated_word_sets
    )

    logger.info(f"Duplicated words: {immediate_duplicated_words}")
    logger.info(f"Repeated sets: {close_repeated_word_sets}")
    system_prompt = (
        f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
        f"Your task is to correct duplicated words in the given text line, ensuring the corrected line retains the "
        f"original meaning and style of 18th-century documents."
        f"\n\n## Guidelines:"
        f"\n1. Identify and correct any duplicated words in the text line."
        f"\n2. Maintain the original meaning and style of the text."
        f"\n3. Ensure corrections accurately reflect the 18th-century language and conventions."
        f"\n4. Just leave one occurrence of the duplicate word, don't delete everything."
        f"\n5. If it is the same word and one is capitalised and one is lowercase, delete one."
        f"\n\n## Duplicated words detected in the text line:"
        f"\n{duplicated_words_part}"
        f"\n\n## Repeated word sets detected in the text line:"
        f"\n{repeated_sets_part}"
        f"\nBased on the guidelines, please analyze the following text line and"
        f" provide the corrected version: {text_line} "
        f"[/INST]\nThen corrected text line is:"
    )

    tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 25
    response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
    raw_response = response[0]['generated_text']
    # print("Raw response:", raw_response)  # Debug print

    json_output_marker = "Then corrected text line is:"

    if json_output_marker in raw_response:
        corrected_text = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
        corrected_text = re.sub(r'\[.*?\]', '', corrected_text)  # Remove any remaining [INST] or similar tags
        corrected_text = corrected_text.strip()  # Trim any surrounding whitespace
        if len(corrected_text) < (len(text_line) / 2):
            corrected_text = text_line
    else:
        corrected_text = text_line

    return corrected_text


def check_and_correct_text_line(text_line, train_set_lines, pipe, mistral_tokenizer):
    logger.debug(f"Checking and correcting text line: {text_line}")

    # Correct misrecognized characters first
    # misrecognized_errors = check_misrecognized_characters(text_line, pipe, mistral_tokenizer)
    # if misrecognized_errors.get("response") == "Yes":
    suggestions = detect_errors_and_suggest_corrections(text_line, train_set_lines)
    corrected_text = correct_text_with_context(text_line, suggestions, pipe, mistral_tokenizer)
        # corrected_text = correct_misrecognized_characters(corrected_text, pipe, mistral_tokenizer)
    # else:
    #     corrected_text = text_line

    # duplicated_words_info = detect_duplicated_words_prompt(corrected_text, pipe, mistral_tokenizer)
    # if duplicated_words_info.get("response") == "Yes":
    #     logger.info(f"Detected duplicated words in '{text_line}'")
    corrected_text = correct_duplicated_words_prompt(corrected_text, pipe, mistral_tokenizer)
    logger.info(f"Text after correcting duplicated words: '{corrected_text}'")

    return corrected_text


def evaluate_test_data_mistral7B(loaded_data, train_set_lines, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        corrected_text_line = check_and_correct_text_line(ocr_text, train_set_lines, pipe, mistral_tokenizer)
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
