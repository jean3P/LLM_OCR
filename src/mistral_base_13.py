import json
import os
import time
import re
from collections import defaultdict
from difflib import get_close_matches, SequenceMatcher

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


# Extract text lines from training data

# Check if a word contains mostly digits
def is_mostly_numeric(word):
    digits_count = sum(char.isdigit() for char in word)
    return digits_count > len(word) / 2


# Function to find the top 3 most similar words, excluding words that are mostly numbers and removing duplicates
def find_top_3_processed_similar_words(word, train_set_lines, is_start_of_line=False):
    def split_suggestion(suggestion, original):
        # Preserve any ending characters from the original word
        if original and suggestion and not suggestion.endswith(original[-1]):
            suggestion += original[-1]

        if is_start_of_line and not suggestion.startswith(original):
            return original

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

    # Remove the original word from matches
    unique_matches = [match for match in unique_matches if match != word]

    split_matches = [split_suggestion(match, word) for match in unique_matches]

    # Save suggestions in memory if they are different from the original word
    if split_matches:
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
    # Return the same text line if it ends with an underscore
    if (
            ocr_text.startswith('\"') and ocr_text.endswith('\"') or
            ocr_text.endswith('_') or
            ocr_text.endswith('=') or
            (ocr_text.count('(') == 1 and ocr_text.count(')') == 0) or
            (ocr_text.count(')') == 1 and ocr_text.count('(') == 0) or
            '&' in ocr_text
    ):
        return ocr_text
    suggestion_part = "\n".join(
        f"Original word from the text line: {ocr_word}, Suggestions for corrections: {', '.join(set(similar_words))}"
        for ocr_word, similar_words in suggestions if similar_words and set(similar_words) != {ocr_word}
    )

    if not suggestion_part:
        suggestion_part = "No suggestions available."

    system_prompt = (
        f"[INST] Act as an 18th-century document analyst specializing in OCR error correction. "
        f"Your task is to correct OCR errors in words or numbers taking into account the suggestions of similar words "
        f"for corrections with a high priority in 18th-century documents (without adding new content)."
        f"Correction does not consist of deleting words but of replacing them with words that are more correct in the "
        f"context of the of the text line"
        f"\n\n## Guidelines:"
        f"\n1. Ensure corrections accurately reflect the 18th-century language and conventions"
        f"\n2. Be sure to keep punctuation marks of the original text line and do not add new punctuation that is "
        f"not in the original line of text"
        f"\n3. Preserve original word splits or cuts (e.g., 'incomple-' should not be combined into 'incomplete')"
        f"\n4. If the original text line is hyphenated, you have to keep the hyphen "
        f"\n5. Don't delete words and dont add words at the end, keep the same number of words from the original text "
        f"line"
        f"\n6. Words cut off at the end with a hyphen must not be completed and do not add contenct at the end of the "
        f"text line"
        f"\n7. Do not modify the end of the text line by adding new content"
        f"\n8. Do not add punctuation mark at the end if the original text line does not have it"
        f"\n9. Do not change proper names even if they are not common"
        f"\n10. Do not delete words like: or, and"
        f"\n11. If the tex line contains something like this: ( k ) must be (k)"
        f"\n12. If the text line starts with ':', do not delete it"
        f"\n\n## Suggestions of similar words of the training set:"
        f"\n{suggestion_part}"
        f"\nYour task is to replace the words with OCR errors based on the guidelines and suggestions correct "
        f"the text line: {ocr_text} [/INST]"
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
    corrected_text = corrected_text.replace(' "', '').strip()
    corrected_text = corrected_text.replace(' \"', '').strip()
    # Remove the specific ' [' character sequence
    corrected_text = corrected_text.replace('ORR]', '')
    if not ']' in corrected_text:
        corrected_text = corrected_text.replace(' [', '')
    corrected_text = re.sub(r'\s+', ' ', corrected_text)
    # Remove only leading and trailing single quotation marks
    if corrected_text.startswith("'") and corrected_text.endswith("'"):
        corrected_text = corrected_text[1:-1].strip()

    # Ensure the length is within reasonable bounds
    if len(corrected_text) > (len(ocr_text) * 1.11):
        corrected_text = ocr_text
    elif len(corrected_text) < (len(ocr_text) / 1.5):
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


# def detect_immediate_repeated_punctuation(text_line):
#     """
#     This function takes a text line as input and returns a list of immediately repeated punctuation marks
#     (like ;,-:) found in the text line. This includes cases where there is whitespace or other characters
#     between the repeated punctuation marks.
#     """
#     # Define the punctuation marks you want to check for repetition
#     punctuation_marks = r'[;,:\-]'
#
#     # Match punctuation marks that are repeated with or without whitespace in between
#     repeated_punctuation = re.findall(r'({0})[\s\W]*\1'.format(punctuation_marks), text_line)
#
#     return repeated_punctuation


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def detect_similar_immediate_repeated_words(text_line, similarity_threshold=0.8):
    """
    This function takes a text line as input and returns a list of similar immediately repeated words found in the text line.
    The function detects words that are at least `similarity_threshold` similar and are immediately adjacent.
    """
    # Tokenize the line into words
    words = re.findall(r'\b\w+\b', text_line)

    # Store pairs of similar words
    similar_repeated_words = []

    # Compare each word with its next immediate word
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        if similar(word1, word2) >= similarity_threshold:
            similar_repeated_words.append((word1, word2))

    return similar_repeated_words


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
    similar_duplicated_words = detect_similar_immediate_repeated_words(text_line)
    # immediate_duplicated_punctuation = detect_immediate_repeated_punctuation(text_line)

    if (not immediate_duplicated_words and not close_repeated_word_sets and not
    similar_duplicated_words):
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

    similar_sets_pair = "\n".join(
        f"Duplicated word: {word_set}" for word_set in similar_duplicated_words
    )

    # similar_sets_duplicated_punctuation = "\n".join(
    #     f"Duplicated punctuation marks: {punctuation_set}" for punctuation_set in immediate_duplicated_punctuation
    # )

    combined_result = "\n".join([duplicated_words_part, similar_sets_pair])
    logger.info(f"Duplicated words: {combined_result}")
    logger.info(f"Repeated sets: {close_repeated_word_sets}")
    # logger.info(f"Duplicated punctuation marks: {immediate_duplicated_punctuation}")

    system_prompt = (
        f"[INST] Act as an 18th-century document analyst specializing in OCR correction. "
        f"Your task is to correct duplicated words in the given text line, ensuring the corrected line retains the "
        f"original meaning and style of 18th-century documents."
        f"\n\n## Guidelines:"
        f"\n1. Identify and correct any duplicated words in the text line"
        f"\n2. Maintain the original meaning and style of the text"
        f"\n3. Ensure corrections accurately reflect the 18th-century language and conventions"
        f"\n4. Just leave one occurrence of the duplicate word, don't delete everything"
        f"\n5. If it is the same word and one is capitalised and one is lowercase, delete one"
        f"\n\n## Duplicated words detected in the text line:"
        f"\n{combined_result}"
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


def check_spelling_in_text_line(original_text_line, pipe, mistral_tokenizer):
    system_prompt = (

        f"[INST] Act as a spelling evaluator. Your task is to analyze the original text line and determine if any "
        f"word is spelled "
        f"incorrectly. Provide your answer as either Yes or No."
        f"\nRespond with either Yes or No only."
        f"\nGiven the original text line: '{original_text_line}'"
        f"[/INST]\nThen the response is:"
    )

    tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 10
    response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
    raw_response = response[0]['generated_text']

    json_output_marker = "Then the response is:"
    spelling_error = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()

    logger.info(f"The text line '{original_text_line}' contains pelling error?: {spelling_error}")

    return spelling_error



def has_misplaced_punctuation(text_line):
    """
    This function checks if a text line contains any misplaced punctuation marks.
    A punctuation mark is considered misplaced if it is not immediately next to the word on its left
    or if there is not exactly one space between the punctuation mark and the word to its right.
    Additionally, it returns False if the text contains an ampersand (&).

    Returns:
        bool: True if there is misplaced punctuation, otherwise False.
    """
    # Return True if the text contains both '(' and ')'
    if '(' in text_line and ')' in text_line and not '[ (' in text_line:
        return True
    elif '&' in text_line or '[' in text_line or '[ (' in text_line:
        return False
    elif text_line.startswith('-') and text_line[1:].strip() and '-' not in text_line[1:]:
        return False

    # Define a pattern to match correct punctuation usage
    correct_punctuation_pattern = r'\w[.,;:?!-]\s\w|\w[.,;:?!-]$'

    # Split the text line into tokens
    tokens = text_line.split()

    for token in tokens:
        # Check if the token contains punctuation
        if re.search(r'[.,;:?!-]', token):
            # If the token doesn't match the correct punctuation pattern, it's misplaced
            if not re.search(correct_punctuation_pattern, token):
                return True

    # If no misplaced punctuation is found, return False
    return False


def check_missing_or_extra_words(original_text, corrected_text):
    """
    This function checks if any word from the original text line is missing in the corrected text line
    or if the corrected text line has extra words, ignoring punctuation marks.
    It maintains the punctuation marks of the corrected text line.

    Args:
        original_text (str): The original text line.
        corrected_text (str): The corrected text line.

    Returns:
        bool: True if any word is missing or if there are extra words in the corrected text line, otherwise False.
    """

    # Remove punctuation from the original text for comparison
    original_words = re.findall(r'\b\w+\b', original_text)

    # Remove punctuation from the corrected text for comparison
    corrected_words = re.findall(r'\b\w+\b', corrected_text)

    # Check if any word in the original text is missing in the corrected text
    for word in original_words:
        if word not in corrected_words:
            return True

    # Check if there are extra words in the corrected text
    for word in corrected_words:
        if word not in original_words:
            return True

    # If all words match and there are no extras, return False
    return False


def check_and_correct_punctuation(text_line, pipe, mistral_tokenizer):
    """
    This function checks if a text line contains a punctuation mark and verifies if it is correctly placed.
    If the punctuation is not correctly placed, it uses the LLM to correct it.
    """
    # Regular expression to find punctuation marks
    # punctuation_pattern = r'[.,;:?!-]'

    # Check if the text line contains punctuation
    is_misplaced = has_misplaced_punctuation(text_line)
    logger.info(f"Is the punctuation mark misplaced?: {is_misplaced}")
    if is_misplaced:
        # Create the system prompt to check punctuation placement
        system_prompt = (
            f"[INST] Your task is to correct only the placement of punctuation marks in the given text line. "
            f"Ensure the punctuation marks are correctly placed close to the left word and with exactly one space "
            f"between the punctuation mark and the word to its right. Do not add any new punctuation or alter the "
            f"original meaning of the text"
            f"\n\n## Guidelines:"
            f"\n1. Punctuation marks should be immediately next to the word on their left"
            f"\n2. Ensure there is exactly one space between the punctuation mark and the word to its right"
            f"\n3. Do not remove any words from the text line "
            f"\n4. Just focus only on misplaced punctuation, if any punctuation is missing don't add it"
            f"\n5. Do not delete this character & from the text line"
            f"\n\nExamples of Corrected Text:"
            f"\n- text line: 'Hello , world' the corrected text line is: 'Hello, world'"
            f"\n- text line: 'Good morning ; everyone' the corrected text line is: 'Good morning; everyone'"
            f"\n- text line: 'Are you ready ? ' the corrected text line is: 'Are you ready? '"
            f"\n\nYour task is only correct the misplaced punctuation marks base on the guidelines and examples, then  "
            f"given the text line: '{text_line}'"
            f"[/INST]\nThen the corrected text line is:"
        )

        tokens_prompt = count_tokens(system_prompt, mistral_tokenizer) + 25
        response = calculate_pipe(pipe, system_prompt, tokens_prompt, 1)
        raw_response = response[0]['generated_text']

        json_output_marker = "Then the corrected text line is:"

        if json_output_marker in raw_response:
            corrected_text = raw_response.split(json_output_marker)[-1].split('\n')[0].strip()
            corrected_text = re.sub(r'\[.*?\]', '', corrected_text)  # Remove any remaining [INST] or similar tags
            corrected_text = corrected_text.replace('[INST', '').replace('[/INST', '').strip()
            corrected_text = corrected_text.replace('[COR]', '').replace('[/COR]', '').strip()
            corrected_text = corrected_text.replace('ORR]', '')
            if not ']' in corrected_text:
                corrected_text = corrected_text.replace(' [', '')
            corrected_text = corrected_text.replace('/', '')
            corrected_text = re.sub(r'\s+', ' ', corrected_text)
            corrected_text = corrected_text.strip()  # Trim any surrounding whitespace

            # Remove only leading and trailing single quotation marks
            # Ensure the length is within reasonable bounds
            if corrected_text.startswith("'") and (corrected_text.endswith("'") or corrected_text.endswith("'.")):
                corrected_text = corrected_text[1:-1].strip()

            if check_missing_or_extra_words(text_line, corrected_text):
                corrected_text = text_line
        else:
            corrected_text = text_line

        logger.info(f"Punctuation checked and corrected for '{text_line}': {corrected_text}")
        return corrected_text
    else:
        # If there are no punctuation marks, return the original text line
        logger.info(f"No punctuation marks found for text: {text_line}")
        return text_line


def check_and_correct_text_line(text_line, train_set_lines, pipe, mistral_tokenizer):
    logger.debug(f"Checking and correcting text line: {text_line}")
    spelling_erros = check_spelling_in_text_line(text_line, pipe, mistral_tokenizer)
    corrected_text = text_line
    if spelling_erros == 'Yes':
        suggestions = suggest_corrections_for_ocr_text(corrected_text, train_set_lines)
        corrected_text = correct_ocr_text_with_suggestions(corrected_text, suggestions, pipe, mistral_tokenizer)
    corrected_text = correct_duplicated_words_in_text_line(corrected_text, pipe, mistral_tokenizer)
    corrected_text = check_and_correct_punctuation(corrected_text, pipe, mistral_tokenizer)
    confidence, justification = evaluate_corrected_text_line_line(text_line, corrected_text, pipe, mistral_tokenizer)
    logger.info(f"Text after correcting duplicated words: '{corrected_text}'")

    return corrected_text, confidence, justification


def evaluate_test_data_mistral7B(loaded_data, train_set_lines, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']

        corrected_text_line, confidence, justification = check_and_correct_text_line(ocr_text, train_set_lines, pipe,
                                                                                     mistral_tokenizer)
        if ocr_text == corrected_text_line:
            cer_mistral = item['cer']
        else:
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
