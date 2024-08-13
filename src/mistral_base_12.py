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
# Dictionary to store suggestions for future use
suggestions_memory = {}
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
def find_top_3_processed_similar_words(word, train_set_lines, is_start_of_line=False):
    def split_suggestion(suggestion, original):
        if original.endswith('-') and not suggestion.endswith('-'):
            return suggestion[:len(original)-1] + '-'
        if is_start_of_line and not suggestion.startswith(original):
            return original
        # Ensure colon preservation
        if ':' in original and ':' not in suggestion:
            return suggestion + ':'
        # Remove punctuation if the original doesn't have it
        if not re.search(r'[^\w\s]', original):
            suggestion = re.sub(r'[^\w\s]', '', suggestion)
        return suggestion

    # Check if we have previously saved suggestions
    if word in suggestions_memory:
        return suggestions_memory[word]

    all_words = [line.split() for line in train_set_lines]
    flat_list = [item for sublist in all_words for item in sublist]

    matches = get_close_matches(word, flat_list, n=3, cutoff=0.85)
    unique_matches = list(dict.fromkeys(matches))  # Remove duplicates while preserving order
    split_matches = [split_suggestion(match, word) for match in unique_matches]

    # Save suggestions in memory if they are different from the original word
    if split_matches and split_matches != [word]:
        suggestions_memory[word] = split_matches

    return split_matches if split_matches else [word]


# Detect OCR Errors and Suggest Corrections
def suggest_corrections_for_ocr_text(ocr_text, train_set_lines):
    suggestions = []
    words = ocr_text.split()
    for idx, word in enumerate(words):
        is_start_of_line = (idx == 0)
        similar_words = find_top_3_processed_similar_words(word, train_set_lines, is_start_of_line)
        suggestions.append((word, similar_words))
    logger.info(f"OCR text '{ocr_text}' suggestions: {suggestions}")
    return suggestions


# Correct Text Line with Context
def correct_ocr_text_with_suggestions(ocr_text, suggestions, pipe, mistral_tokenizer):

    # if all(len(similar_words) == 1 and similar_words[0] == ocr_word for ocr_word, similar_words in suggestions):
    #     return ocr_text

    suggestion_part = "\n".join(
        f"Original word from the text line: {ocr_word}, Suggestions for corrections: {', '.join(set(similar_words))}"
        for ocr_word, similar_words in suggestions if similar_words and set(similar_words) != {ocr_word}
    )

    if not suggestion_part:
        suggestion_part = "No suggestions available."

    system_prompt = (
        f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
        f"Your task is to correct OCR errors in words or numbers taking into account the suggestions of similar words "
        f"for corrections with a high priority in 18th-century documents."
        f"\n\n## Guidelines:"
        f"\n1. Ensure corrections accurately reflect the 18th-century language and conventions"
        # f"\n2. You need to maintain the same number of words of the text line"
        f"\n2. Be sure to keep punctuation marks of the original text line and do not add new punctuation that is "
        f"not in the original line of text"
        f"\n3. Preserve original word splits or cuts (e.g., 'incomple-' should not be combined into 'incomplete')"
        f"\n4. If the original text line is hyphenated, you have to keep the hyphen "
        f"\n5. Don't delete words that are not duplicated"
        f"\n6. Words cut off at the end with a hyphen should not be completed"
        f"\n7. Do not modify the end of the text line by adding new content"
        # f"\n8. Suggestions are used for corrections of words with OCR errors"
        # f"\n9. Learn the corrected words so that you can correct it automatically"
        f"\n\n## Suggestions of similar words of the training set:"
        f"\n{suggestion_part}"
        f"\nBased on the guidelines and suggestions correct the text line: {ocr_text} [/INST]"
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

    # Remove any remaining '[INST]' or '[/INST]' tags manually and ensure text formatting
    corrected_text = result.replace('[INST]', '').replace('[/INST]', '').strip()
    corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
    corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
    corrected_text = corrected_text.replace('/C', '').strip()
    corrected_text = corrected_text.replace('ORRECT]', '').strip()
    corrected_text = corrected_text.replace('ORRECTED]', '').strip()
    # Remove the specific ' [' character sequence
    corrected_text = corrected_text.replace(' [', '')
    corrected_text = re.sub(r'\s+', ' ', corrected_text)

    # Ensure the length is within reasonable bounds
    if len(corrected_text) > (len(ocr_text) * 1.2):
        corrected_text = ocr_text
    elif len(corrected_text) < (len(ocr_text) / 2):
        corrected_text = ocr_text

    logger.info(f"Corrected text for '{ocr_text}': {corrected_text}")
    return corrected_text


def detect_immediate_repeated_words(text_line):
    """
    This function takes a text line as input and returns a list of immediately repeated words found in the text line.
    This includes cases where there is punctuation or whitespace between the repeated words, and insensitive.
    """
    # Match words that are repeated with or without punctuation in between, insensitive
    repeated_words = re.findall(r'\b(\w+)\b[\s\W]+\b\1\b', text_line, flags=re.IGNORECASE)
    # # Ensure the found duplicates are indeed duplicates
    # actual_repeats = [match for match in repeated_words if text_line.lower().count(match.lower()) > 1]
    return repeated_words


def detect_close_repeated_word_sequences(text_line):
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


def correct_duplicated_words_in_text_line(text_line, pipe, mistral_tokenizer):
    # First, use the find_immediate_repeated_words function to detect repeated words
    immediate_duplicated_words = detect_immediate_repeated_words(text_line)
    close_repeated_word_sets = detect_close_repeated_word_sequences(text_line)

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
        corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
        corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
        corrected_text = corrected_text.replace(' [', '')
        corrected_text = corrected_text.replace('/', '')
        corrected_text = re.sub(r'\s+', ' ', corrected_text)
        corrected_text = corrected_text.strip()  # Trim any surrounding whitespace
        if len(corrected_text) < (len(text_line) / 1.8):
            corrected_text = text_line
    else:
        corrected_text = text_line

    return corrected_text


def evaluate_corrected_text_line_line(original_text_line, corrected_text_line, pipe, mistral_tokenizer):
    system_prompt = (
        f"[INST] Act as an 18th-century text line evaluator. Your task is to analyze the original text line by an "
        f"OCR model and evaluate the corrected text line provided by the LLM. Determine if the LLM's corrected text "
        f"line accurately fixes the OCR errors. Measure your confidence in the accuracy of the corrected text line on "
        f"a scale from 0 to 100 and provide a detailed justification for your assessment."
        f"\nProvide the confidence score and the justification as follows:\n"
        f"Confidence: <confidence_score>\nJustification: <justification>"
        f"\nGiven the original text line: '{original_text_line}' and the text line from the LLM: '{corrected_text_line}'"
        f"[/INST]\nThen the confidence and the justification is:"
    )

    tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 100
    response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
    raw_response = response[0]['generated_text']

    json_output_marker = "Then the confidence and the justification is:"
    corrected_text = raw_response.split(json_output_marker)[-1].strip()

    # Extract the confidence and justification
    confidence_marker = "Confidence:"
    justification_marker = "Justification:"

    # Extract confidence
    confidence_section = corrected_text.split(confidence_marker)[-1].strip()
    confidence = confidence_section.split('\n')[0].strip()

    # Extract justification
    justification_section = corrected_text.split(justification_marker)[-1].strip()
    justification_lines = justification_section.split('\n')
    justification = justification_lines[0].strip()  # Take the first line of the justification

    logger.info(f"confidence: {confidence}")
    logger.info(f"justification: {justification}")

    return confidence, justification


def check_and_correct_text_line(text_line, train_set_lines, pipe, mistral_tokenizer):
    logger.debug(f"Checking and correcting text line: {text_line}")
    suggestions = suggest_corrections_for_ocr_text(text_line, train_set_lines)
    corrected_text = correct_ocr_text_with_suggestions(text_line, suggestions, pipe, mistral_tokenizer)
    corrected_text = correct_duplicated_words_in_text_line(corrected_text, pipe, mistral_tokenizer)
    confidence, justification = evaluate_corrected_text_line_line(text_line, corrected_text, pipe, mistral_tokenizer)
    logger.info(f"Text after correcting duplicated words: '{corrected_text}'")

    return corrected_text, confidence, justification


def evaluate_test_data_mistral7B(loaded_data, train_set_lines, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        corrected_text_line, confidence, justification = check_and_correct_text_line(ocr_text, train_set_lines, pipe, mistral_tokenizer)
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
                'confidence': confidence,
                'justification': justification
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
