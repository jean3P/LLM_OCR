import os
import time
import re
import evaluate
import torch
import transformers
from transformers import AutoTokenizer, pipeline

from confidence_calculator import calculate_confidence
from handle_dataset_washington import save_to_json, load_from_json
from utils.constants import results_test_trocr, results_mixed_LLM_MISTRAL, automated_resuts, TOKEN
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")

torch.cuda.empty_cache()
BAD_GRAMMAR = 'No'

GOOD_GRAMMAR = 'Yes'

device = torch.device("cuda")
cer_metric = evaluate.load('cer')

# Global variable to store context sentences for each document
document_contexts = {}


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Ensure all GPUs are visible

def calculate_pipe(pipe, prompt, nummer_length, top_k):
    return pipe(prompt, max_length=nummer_length, do_sample=True, top_k=top_k, num_return_sequences=1,
                pad_token_id=pipe.tokenizer.eos_token_id)


def check_grammar(sentence, pipe):
    prompt = f"Answer Yes or No: Is this sentence grammatically correct? '{sentence}', then the answer is:"

    try:
        nummer_length = 100

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


def check_standardize_terms(sentence, pipe):
    prompt = (f"Answer Yes or No: Does this sentence contain any terms or abbreviations "
              f"that need to be standardized? '{sentence}', then the answer is:")

    try:
        nummer_length = 100  # Adjust based on expected response length

        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)

        response = corrected_text[0]['generated_text'].split(', then the answer is:')[-1].strip()
        if 'Yes' in response:
            standardization_needed = 'Yes'
        elif 'No' in response:
            standardization_needed = 'No'
        else:
            standardization_needed = 'Error'
    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        standardization_needed = 'Error'

    return standardization_needed


def check_spelling(sentence, pipe):
    prompt = f"Answer Yes or No: Does this sentence contain any spelling mistakes? '{sentence}', then the answer is:"

    try:
        nummer_length = 100  # Adjust as necessary based on expected response length

        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)

        response = corrected_text[0]['generated_text'].split(', then the answer is:')[-1].strip()
        if 'Yes' in response:
            spelling_check_results = 'Yes'
        elif 'No' in response:
            spelling_check_results = 'No'
        else:
            spelling_check_results = 'Error'
    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        spelling_check_results = 'Error'

    return spelling_check_results


def correct_sentence_using_examples(sentence, context, pipe, short_system=False, ground_label=''):
    # if short_system:
    nummer_length = (len(sentence) * 2) + 2200
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
        "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
        "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
        "\n- The confidence percentage is calculated based on your trust. "
        "\n\n## Examples of corrected sentences modelling cases"
        "\n1. OCR Error from the User: 288. Letters Orders and Instructions Yuemler, "
        "   The JSON output is:"
        "     Corrected sentence: 308th Letters, Orders, and Instructions, December 1755."
        "     Percentage of confidence: 77"
        "     Justification: Corrected the year misrecognition 'Yuemler' to 'December' and adjusted the incorrect number '288' to '308'. The relatively high confidence indicates that parts of the text were recognized correctly despite the errors."
        "\n2. OCR Error from the User: the Thus, We cand to be under the under the same directions "
        "   The JSON output is:"
        "     Corrected sentence: the Stores, Vc. and to be under the same directions"
        "     Percentage of confidence: 77"
        "     Justification: Mistral needs to eliminate redundant phrases like 'under the' repeated and correct 'Thus, We cand' to 'the Stores, Vc.' despite lower confidence, indicating multiple recognition errors."
        "\n3. OCR Error from the User: of Octopus, Shoes, Stationy, Startings, Ibids, Visits, to. Preparationa- "
        "   The JSON output is:"
        "     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona-"
        "     Percentage of confidence: 90"
        "     Justification: This case shows significant text alteration; 'Octopus' is a stark deviation from 'Clothes' and other elements like 'Stationy' to 'Stocking'. It illustrates the challenge of correcting highly erroneous OCR outputs."
        "\n4 OCR Error from the User: thes, Vc. and to me directions "
        "   The JSON output is:"
        "     Corrected sentence: the Stores, Vc., and to be under the same directions"
        "     Percentage of confidence: 90"
        "     Justification: The word 'thes' is replaced with 'the Stores' as it seems to be a contextually appropriate correction. The addition of 'to be' after '&c.' improves the sentence structure."
        "\n5 OCR Error from the User: Alexamatic: December: December 6. 1755. "
        "   The JSON output is:"
        "     Corrected sentence: Alexandria: December 16th. 1755."
        "     Percentage of confidence: 90"
        "     Justification: Mistral would need to correct multiple date errors and revert 'Alexamatic' back to 'Alexandria', reflecting challenges in recognizing proper nouns and dates correctly."
        "\n6 OCR Error from the User: remain her until the animal of the quiet with, "
        "   The JSON output is:"
        "     Corrected sentence: remain here until the arrival of the vessel with"
        "     Percentage of confidence: 80"
        "     Justification: Correction from 'animal' to 'arrival' and 'quiet' to 'vessel' suggests contextual awareness, while 'her' to 'here' indicates a need for attention to detail in similar sounding words."
        "\n7 OCR Error from the User: as below ordered. To soon as the Stores arrive, arrive, you "
        "   The JSON output is:"
        "     Corrected sentence: as before ordered. So soon as the Stores arrive, you"
        "     Percentage of confidence: 79"
        "     Justification: Redundancy in 'arrive, arrive,' and errors like 'below' for 'before' require understanding of context and basic syntax corrections."
        "\n8 OCR Error from the User: fcient number number of Uruguay to carry them them to thin- "
        "   The JSON output is:"
        "     Corrected sentence: ficient number of waggons to carry them to Win-"
        "     Percentage of confidence: 95"
        "     Justification: Substantial errors such as 'Uruguay' instead of 'waggons' and repeated 'them' showcase a need for semantic corrections and reduction of redundancy."
        "\n9 OCR Error from the User: of Cophys, Stations, Vc. proportion- "
        "   The JSON output is:"
        "     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona-"
        "     Percentage of confidence: 95"
        "     Justification: Misinterpretation of entire phrases indicates a requirement for improvements in recognition of related items and terminology (e.g., 'Cophys' to 'Clothes')."
        "\n10 OCR Error from the User: present is made. If your men can not be "
        "   The JSON output is:"
        "     Corrected sentence: zlement is made. If your men can not be"
        "     Percentage of confidence: 75"
        "     Justification: Minor errors such as 'present' instead of 'zlement' highlight the need for precise character recognition to improve overall text accuracy."
        "\n11 OCR Error from the User: asict Guard over, that no amles- "
        "   The JSON output is:"
        "     Corrected sentence: a strict Guard is kept over them, that no embez-"
        "     Percentage of confidence: 65"
        "     Justification: The prediction misses several words and misinterprets others, such as 'asict' for 'a strict' and 'amles' for 'embez'. Correcting these requires understanding phrase structure and word boundaries."
        "\n12 OCR Error from the User: concerning practiced on theired on their march up; there "
        "   The JSON output is:"
        "     Corrected sentence: conveniently Quartered on their march up; there"
        "     Percentage of confidence: 95"
        "     Justification: This error includes adding extra words and changing 'Quartered' to a nonsensical phrase. Mistral needs to focus on maintaining the integrity of phrases and eliminating unnecessary insertions."
        "\n13 OCR Error from the User: should from this place - it appear "
        "   The JSON output is:"
        "     Corrected sentence: should desert from this place, - it would be proper"
        "     Percentage of confidence: 87"
        "     Justification: Major parts of the sentence are missing or altered. Mistral's correction needs to focus on recovering missing verbs and adjusting punctuation for clarity."
        "\n14 OCR Error from the User: cane to purchase no more more than is absolutely "
        "   The JSON output is:"
        "     Corrected sentence: care to purchase no more than is absolutely"
        "     Percentage of confidence: 95"
        "     Justification: Redundancy in 'no more more' and incorrect word 'cane' for 'care' show a need for word choice refinement and error detection in repeated terms."
        "\n15 OCR Error from the User: necessary, for the support and relief and relief of the Regional"
        "   The JSON output is:"
        "     Corrected sentence: necessary, for the support and relief of the Regiment"
        "     Percentage of confidence: 75"
        "     Justification: Repeated phrases and incorrect terms like 'Regional' for 'Regiment' require attention to context and specific terminology associated with the text's subject matter."
        "\n16 OCR Error from the User: should deliver from this place, if would be proper "
        "   The JSON output is:"
        "     Corrected sentence: should desert from this place, - it would be proper"
        "     Percentage of confidence: 85"
        "     Justification: TIncorrect verb 'deliver' should be 'desert,' and the connection phrase 'if' should be corrected to '- it.' This example emphasizes the importance of verb accuracy and maintaining grammatical structures."
        "\n17 OCR Error from the User: to. To Doctor James I ams Gravity of the Virginia "
        "   The JSON output is:"
        "     Corrected sentence: 16th. To Doctor James Craik, of the Virginia"
        "     Percentage of confidence: 83"
        "     Justification: This error includes misrecognition of names and unnecessary repetitions, showing a need for improved name recognition and elimination of redundant elements."
        "\n18 OCR Error from the User: Given at. Alexandria Ye. "
        "   The JSON output is:"
        "     Corrected sentence: Given at Alexandria Vc."
        "     Percentage of confidence: 95"
        "     Justification: Mistaken period after 'at' and incorrect 'Ye' for 'Vc' highlight the need for attention to punctuation and abbreviation accuracy in OCR processes."
        "\n19 OCR Error from the User: G.W. Aidd Gc.com. "
        "   The JSON output is:"
        "     Corrected sentence: G.W. aid de camp."
        "     Percentage of confidence: 97"
        "     Justification: The prediction inaccurately expands abbreviations and adds unnecessary elements, indicating a need for better understanding of common abbreviations and reducing overcorrection."
        "</s>"
    )

    adaptation_request = (
        f"<s>"
        f"[INST] Based on the guidelines and illustrated examples, accurately correct the OCR errors in the following "
        f"sentence: '{sentence}'.[/INST]</s>"
    )

    prompt = f"{system_prompt}\n{adaptation_request}\nThen the output in JSON format is:"
    # nummer_length = (len(sentence) * 2) + 2100

    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        response = corrected_text[0]['generated_text'].split('Then the output in JSON format is:')[-1].strip()
        response = response.split('\n')[0].strip()


    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = sentence
    return response


def get_document_id(file_name):
    # Extract the first 3 digits of the file_name as the document ID
    return file_name.split("-")[0]


def correct_sentences_1(sentence_data, pipe, batch_size=10, short_system=False):
    global document_contexts

    start = time.time()
    corrected_sentence = ''
    confidence = ''
    justification = ''
    for i in range(0, len(sentence_data), batch_size):
        batch = sentence_data[i:i + batch_size]
        for data in batch:
            sentence = data['sentence']
            file_name = data['file_name']
            ground_truth_label = data['ground_truth_label']
            document_id = get_document_id(file_name)

            if document_id not in document_contexts:
                document_contexts[document_id] = []

            processed_sentence = correct_sentence_using_ocr_errors(sentence, '', pipe, short_system, ground_truth_label)
            # print(processed_sentence)
            if processed_sentence != 'Error':
                # Extracting the actual corrected sentence from the output
                match_sentence = re.search(r"Corrected sentence: (.+?)\s+Percentage of confidence:", processed_sentence)
                if match_sentence:
                    corrected_sentence = match_sentence.group(1)  # The actual sentence text
                    # corrected_sentence = corrected_sentence.replace("'", "")
                else:
                    corrected_sentence = sentence  # Fallback text if no sentence is found

                # Extracting confidence and justification
                match_details = re.search(r'Percentage of confidence: (\d+).*?Justification: (.+)', processed_sentence)
                if match_details:
                    confidence = match_details.group(1)
                    justification = match_details.group(2)
                else:
                    confidence = '0'
                    justification = 'No justification provided, sentence is returned uncorrected.'
            else:
                corrected_sentence = sentence  # return original if error
                confidence = '0'
                justification = 'Sentence is returned uncorrected.'

    end = time.time()
    print(f"Time taken: {end - start} seconds, "
          f"Corrected Sentence: {corrected_sentence} | Confidence: {confidence} | Justification: {justification} ")
    return corrected_sentence, confidence, justification


def correct_sentences_2(sentence_data, pipe, batch_size=10, short_system=False):
    global document_contexts

    start = time.time()
    corrected_sentence = ''
    confidence = ''
    justification = ''
    for i in range(0, len(sentence_data), batch_size):
        batch = sentence_data[i:i + batch_size]
        for data in batch:
            sentence = data['sentence']
            file_name = data['file_name']
            ground_truth_label = data['ground_truth_label']
            document_id = get_document_id(file_name)

            if document_id not in document_contexts:
                document_contexts[document_id] = []

            processed_sentence = correct_sentence_using_examples(sentence, '', pipe, short_system, ground_truth_label)
            # print(processed_sentence)
            if processed_sentence != 'Error':
                # Extracting the actual corrected sentence from the output
                match_sentence = re.search(r"Corrected sentence: (.+?)\s+Percentage of confidence:", processed_sentence)
                if match_sentence:
                    corrected_sentence = match_sentence.group(1)  # The actual sentence text
                    # corrected_sentence = corrected_sentence.replace("'", "")
                else:
                    corrected_sentence = sentence  # Fallback text if no sentence is found

                # Extracting confidence and justification
                match_details = re.search(r'Percentage of confidence: (\d+).*?Justification: (.+)', processed_sentence)
                if match_details:
                    confidence = match_details.group(1)
                    justification = match_details.group(2)
                else:
                    confidence = '0'
                    justification = 'No justification provided, sentence is returned uncorrected.'
            else:
                corrected_sentence = sentence  # return original if error
                confidence = '0'
                justification = 'Sentence is returned uncorrected.'

    end = time.time()
    print(f"Time taken: {end - start} seconds, "
          f"Corrected Sentence: {corrected_sentence} | Confidence: {confidence} | Justification: {justification} ")
    return corrected_sentence, confidence, justification


def correct_sentence_using_ocr_errors(sentence, context, pipe, short_system=False, ground_label=''):
    nummer_length = (len(sentence) * 2) + 3700
    system_prompt = (
        "<s>"
        "[INST] Your primary task is to meticulously correct OCR (Optical Character Recognition) errors in a "
        "collection of 18th-century documents. These documents contain a variety of errors, ranging from simple "
        "misspellings to more complex issues like incorrect abbreviations and misinterpretations of terms. "
        "Your corrections must strive for precision, preserving the authenticity and integrity of the original "
        "manuscripts. It's imperative to avoid introducing new information or excluding essential details. "
        "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
        "linguistic conventions of the 18th century.[/INST]"
        f"\n\nSubstitution Errors:"
        f"\n- '8' -> '1', 't'"
        f"\n- '.' -> ':', 'o', 'l', 't', 'e', ' ', 's', 'i', 'd', 'm', 'c', 'f', '7', 'n', 'W', 'L', 'w', 'h', '1', 'a', 'D', 'T', '8', 'r', '5', 'b', ','"
        f"\n- ' ' -> 'l', 'B', '.', 'i', 'd', 'm', 'f', 'n', ';', 'W', 'w', 'h', 'p', 'g', 'I', 'T', 'r', 'y', 'b', ',', 'N', ':', 'o', 'k', 't', '-', 'e', 's', 'R', 'c', '7', 'H', 'u', 'a', '1', 'S', 'D', 'C', 'v'"
        f"\n- 'L' -> ' '"
        f"\n- 'e' -> 'l', 'z', '.', 'G', 'i', 'd', 'm', 'f', 'n', ';', 'W', 'h', 'w', 'p', 'g', 'V', 'r', 'y', '5', 'b', ',', 'o', 't', '-', ' ', 's', 'R', 'c', '7', 'A', 'u', 'L', 'q', 'a', 'D', 'v', 'S'"
        f"\n- 't' -> 'o', 'l', 'B', '.', '-', 'e', ' ', 's', 'i', 'd', 'm', 'c', 'f', 'x', 'H', 'A', 'n', 'u', 'W', 'h', 'w', 'L', 'p', 'g', 'I', 'a', 'v', 'r', 'y', '?', '5', 'b', 'S', ','"
        f"\n- 'r' -> ':', 'o', 'l', 'k', 't', 'z', '.', 'e', ' ', 's', 'i', 'J', 'm', 'd', 'c', 'f', 'R', 'n', 'u', 'h', 'w', 'p', 'g', 'O', 'I', 'a', 'T', 'C', 'y', 'v', ','"
        f"\n- 's' -> 'o', 'l', 'z', 't', 'B', '-', 'e', ' ', 'i', 'd', 'm', 'U', 'x', 'c', 'f', 'n', 'u', 'w', 'h', 'p', 'g', 'I', 'a', 'V', 'r', 'y', 'T', 'b', 'S', ','"
        f"\n- 'O' -> 'a', ' '"
        f"\n- 'd' -> 'l', 'o', 't', '.', 'e', ' ', 's', 'i', 'f', 'c', 'n', 'u', 'h', 'w', 'p', 'g', 'a', 'C', 'r', 'Q', 'b', 'v', ','"
        f"\n- 'a' -> 'l', 'o', 'k', 't', '.', '-', 'e', 's', ' ', 'i', 'd', 'm', 'f', 'x', 'c', 'A', 'n', 'u', 'w', 'h', 'p', 'g', 'V', 'r', 'y', 'C', 'v', 'b', 'S', ','"
        f"\n- 'n' -> 'o', 'l', 't', '.', '-', 'e', 's', ' ', 'i', 'J', 'm', 'd', 'c', 'f', 'u', 'W', 'h', 'w', 'p', 'g', 'O', 'I', 'a', 'D', 'V', 'r', 'y', 'v', '_', 'b', 'S', ',', 'N'"
        f"\n- 'I' -> 't', 'f', 'L', 'e', ' ', 'h'"
        f"\n- 'u' -> 'l', 'o', 't', '-', 'e', 's', ' ', 'i', 'd', 'm', 'R', 'c', 'f', 'n', ';', 'w', 'h', 'p', 'a', 'r', 'y', 'j', 'b', 'S', ','"
        f"\n- 'c' -> 'o', 'l', 't', 'M', '.', 'e', ' ', 's', 'G', 'i', 'd', 'm', 'x', 'f', 'n', 'u', 'W', 'h', '(', 'w', 'g', 'Y', 'O', 'a', '1', 'D', 'V', 'r', 'C', '5', 'b', 'N'"
        f"\n- 'i' -> 'l', 'o', 't', '.', '-', 'e', 's', ' ', 'G', 'd', 'm', 'R', 'c', 'f', 'n', ';', 'u', 'W', 'h', 'w', '6', 'p', 'g', 'I', 'a', 'V', 'r', 'y', 'v', ','"
        f"\n- 'o' -> 'l', 'k', 't', '.', '-', 'e', ' ', 's', 'i', 'd', 'm', 'f', 'c', 'n', 'u', 'W', 'h', 'w', 'p', 'g', 'O', 'a', 'D', 'V', 'r', 'y', 'v', 'C', 'T', 'b', 'S', ',', 'F'"
        f"\n- 'D' -> '6', 'o', '1', 'e', 's', ' ', '8', 'r', 'v', 'd', 'c', 'n', 'u'"
        f"\n- 'm' -> 'o', 'l', 'k', 'M', 't', '.', '-', 'e', 's', ' ', 'i', 'd', 'J', 'n', 'w', 'h', 'p', 'g', 'a', 'D', 'r', 'y', '5', 'b', ','"
        f"\n- 'b' -> 'o', 'l', 'k', 't', 'e', ' ', 's', 'i', 'd', 'm', 'A', 'n', 'u', 'w', 'h', 'a', 'r', 'y', '5'"
        f"\n- '1' -> '6', 'M', 't', 'e', ' ', 'r', 'm', 'c', '7', '5', 'b', 'N'"
        f"\n- '7' -> 'c', 'r', '5', '1', 'e', ' ', 'b'"
        f"\n- '5' -> '.', '1', 'e', 's', ' ', 'r', 'm', '7', 'b'"
        f"\n- 'h' -> 'o', 'k', 'l', 't', '.', 'e', ' ', 's', 'i', 'd', 'm', 'J', 'c', 'f', 'n', 'u', 'W', 'w', 'p', 'a', 'T', 'r', 'y', '5', 'b', 'v', ','"
        f"\n- 'l' -> 'o', 'B', 't', 'M', '.', 'e', ' ', 's', 'i', 'd', 'm', 'f', 'c', 'A', 'n', 'u', 'w', 'h', 'p', 'a', 'r', 'y', 'b', ','"
        f"\n- 'v' -> 'o', 'k', 'l', 't', 'a', 'e', ' ', 's', 'r', 'y', 'i', 'V', 'c', 'h', 'n', 'W', 'w'"
        f"\n- 'S' -> 'o', 'l', 'K', 't', '.', 'e', ' ', 'T', 'r', 'i', 'c', 'b', ',', 'u', 'h'"
        f"\n- ',' -> 'o', 't', '-', 'e', ' ', 's', 'i', 'c', 'n', ';', 'u', 'w', 'h', 'p', 'a', 'r', 'y', 'v'"
        f"\n- 'y' -> 'o', 'l', 'B', 't', '.', 'e', ' ', 'i', 'd', 'f', 'n', 'u', 'w', 'p', 'a', 'Q', 'r', 'b', 'v', ','"
        f"\n- 'p' -> 'o', 'z', 't', 'e', ' ', 'G', 'd', 'm', 'x', 'n', 'u', 'h', 'a', 'Q', 'r', 'y', 'C', ','"
        f"\n- 'f' -> 'o', 'l', 't', '.', 'e', ' ', 'i', 'm', 'n', 'u', 'h', 'p', 'a', 'T', 'r', 'C', 'y', 'b'"
        f"\n- '-' -> 'p', 't', '.', 'I', 'a', 'e', ' ', 'V', 'i', 'd', 'c', 'b', 'n', 'h'"
        f"\n- 'w' -> 'o', 'l', 't', 'a', 'e', 's', ' ', 'r', 'i', 'm', 'f', 'W', ',', 'N', 'u', 'h'"
        f"\n- 'g' -> 'o', 't', 'e', 's', ' ', 'r', 'm', 'f', ',', 'b', 'S', 'n', 'u', 'h'"
        f"\n- 'W' -> 'o', 'M', 't', 'e', ' ', 's', 'r', 'i', 'c', 'n'"
        f"\n- ':' -> 'd', 't', '.', 'W', 'n', 'e', ' ', 'r'"
        f"\n- 'x' -> ':', 'i', 'l', 'c', ';', 'a', 'e'"
        f"\n- 'k' -> 't', '.', 'a', 's', ' ', 'r', 'G', 'i', ';', 'u', 'h'"
        f"\n- 'V' -> 'o', 'O', 'D', ' ', 'r', 'b', 'S', 'W'"
        f"\n- 'T' -> 'l', 'K', 'B', 't', '.', 'e', ' ', 'f', 'W', 'h'"
        f"\n- 'z' -> 'o', 'l', 'p', 'M', ' ', 'v', 'c', 'S', ',', 'h'"
        f"\n- 'Q' -> 'a', 'o', 'r'"
        f"\n- '6' -> 'o', '0', 't', '7', '5'"
        f"\n- 'J' -> 'r', 'n', 'm', 'h'"
        f"\n- 'C' -> 'o', 'l', 'B', 'p', 't', 'O', 'a', 'e', 's', ' ', 'E', 'm', 'c', 'h'"
        f"\n- 'H' -> 'N'"
        f"\n- ';' -> 'o', 't', ' ', 'r', 'm', 'h', ',', 'u', 'w'"
        f"\n- 'R' -> 'o', 't', 'e', ' ', 'i', 'd', 'm', 'b', 'h'"
        f"\n- '3' -> 'S', '2'"
        f"\n- '0' -> '.', 'c', 'e', 'D', '8'"
        f"\n- '9' -> 'G', 'c', 'O'"
        f"\n- 'A' -> 't', 'f', 'I', 'n', 'e', ' '"
        f"\n- 'M' -> 'n', 'h'"
        f"\n- 'N' -> 'M', 'V'"
        f"\n- '2' -> '.', 'e', 'D', ' ', '8', 'r', 'b'"
        f"\n- 'E' -> 'a', 'l', 'u'"
        f"\n- 'j' -> 'i', 'n', ' '"
        f"\n- 'G' -> 'B', 't', 'M', 'S', 'r'"
        f"\n- 'P' -> 'I', 'a', 'u', 'T'"
        f"\n- 'B' -> 'o', 't', 's'"
        f"\n- 'F' -> 'e', ' '"
        f"\n\nInsertion Errors:"
        f"'m', 'g', 'l', 'e', 'c', '5', 'V', ';', 't', 'u', 'x', 'v', '6', 'i', '8', 'S', 'y', 'k', ' '"
        f" 'n', 'h', ':', 'D', 'a', 'R', 'd', '7', '2', 'r', 's', '-', '1', 'p', 'z', ',', 'W', 'b'"
        f" 'C', 'o', 'f', '.'"
        "\n\nDeletion Errors:"
        f"'l', ',', 'u', '7', 'h', 'S', 'd', '.', 'a', 'c', 'e', 'y', 's', 'n', 'z', 'r', '2', 'p', 'f'"
        f" '-', ' ', 'i', 'C', 'm', 't', 'H', '0', 'b', '1', '5', 'V', 'o', 'w', '6', 'v', 'R', ';', 'g'"
        "\n\n## Examples of corrected sentences modelling cases"
        "\n1. OCR Error from the User: 288. Letters Orders and Instructions Yuemler, "
        "   The JSON output is:"
        "     Corrected sentence: 308th Letters, Orders, and Instructions, December 1755."
        "     Percentage of confidence: 77"
        "     Justification: Corrected the year misrecognition 'Yuemler' to 'December' and adjusted the incorrect number '288' to '308'. The relatively high confidence indicates that parts of the text were recognized correctly despite the errors."
        "\n2. OCR Error from the User: the Thus, We cand to be under the under the same directions "
        "   The JSON output is:"
        "     Corrected sentence: the Stores, Vc. and to be under the same directions"
        "     Percentage of confidence: 77"
        "     Justification: Mistral needs to eliminate redundant phrases like 'under the' repeated and correct 'Thus, We cand' to 'the Stores, Vc.' despite lower confidence, indicating multiple recognition errors."
        "\n3. OCR Error from the User: of Octopus, Shoes, Stationy, Startings, Ibids, Visits, to. Preparationa- "
        "   The JSON output is:"
        "     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona-"
        "     Percentage of confidence: 90"
        "     Justification: This case shows significant text alteration; 'Octopus' is a stark deviation from 'Clothes' and other elements like 'Stationy' to 'Stocking'. It illustrates the challenge of correcting highly erroneous OCR outputs."
        f"</s>"
    )
    adaptation_request = (
        f"<s>"
        f"[INST] Based on the substitution errors, insertion errors, deletion errors, accurately correct the OCR errors "
        f"in the following sentence: '{sentence}'.[/INST]</s>"
    )
    prompt = (f"{system_prompt}\n{adaptation_request}\nThen the output in JSON format is:")

    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        response = corrected_text[0]['generated_text'].split('Then the output in JSON format is:')[-1].strip()
        response = response.split('\n')[0].strip()
        # print(response)
    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = sentence
    return response


def check_label_correctness(sentence, pipe):
    nummer_length = 800
    prompt = (
        f"<s>[INST] You will be provided with a sentence, and you need to respond with 'Yes' or 'No' to the question: Is the provided sentence, that comes from an OCR model, correct?\n"
        f"If you notice that it has OCR errors, then the sentence is not correct.\n\n"
        f"Examples:\n"
        f"1. Sentence: 288. Letters Orders and Instructions Yuemler,\n"
        f"   Response: No\n"
        f"2. Sentence: 308th Letters, Orders, and Instructions, December 1755.\n"
        f"   Response: Yes\n"
        f"3. Sentence: the Thus, We cand to be under the under the same directions\n"
        f"   Response: No\n"
        f"4. Sentence: the Stores, Vc. and to be under the same directions\n"
        f"   Response: Yes\n"
        f"3. Sentence: of Octopus, Shoes, Stationy, Startings, Ibids, Visits, to. Preparationa-\n"
        f"   Response: No\n"
        f"4. Sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona-\n"
        f"   Response: Yes\n"
        f"5. Sentence: thes, Vc. and to me directions\n"
        f"   Response: No\n"
        f"6. Sentence: the Stores, Vc., and to be under the same directions\n"
        f"   Response: Yes\n"
        f"7. Sentence: Alexamatic: December: December 6. 1755.\n"
        f"   Response: No\n"
        f"8. Sentence: Alexandria: December 16th. 1755.\n"
        f"   Response: Yes\n"
        f"9. Sentence: remain her until the animal of the quiet with,\n"
        f"   Response: No\n"
        f"10. Sentence: remain here until the arrival of the vessel with\n"
        f"   Response: Yes\n"
        f"11. Sentence: as below ordered. To soon as the Stores arrive, arrive, you\n"
        f"   Response: No\n"
        f"12. Sentence: as before ordered. So soon as the Stores arrive, you\n"
        f"   Response: Yes\n"
        f"13. Sentence: fcient number number of Uruguay to carry them them to thin-\n"
        f"   Response: No\n"
        f"14. Sentence: ficient number of waggons to carry them to Win-\n"
        f"   Response: Yes\n"
        f"15. Sentence: of Cophys, Stations, Vc. proportion-\n"
        f"   Response: No\n"
        f"16. Sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona-\n"
        f"   Response: Yes\n"
        f"Given the sentence: <input>{sentence}</input>\n\n"
        f"Please respond with 'Yes' or 'No' based on the correctness of the sentence.[/INST]</s>\n"
        f"Then the respond is:"
    )
    start = time.time()
    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        if corrected_text and 'generated_text' in corrected_text[0]:
            response = corrected_text[0]['generated_text'].strip()
            # Debugging output
            # print(f"Raw response: {response}")

            # Extract the part after "Please respond with 'Yes' or 'No' based on the correctness of the sentence."
            response = response.split("Then the respond is:")[-1].strip()
            # print("===========")
            # print(f"Raw response: {response}")
            # Extract the first "Yes" or "No"
            if 'Yes' in response:
                answer = 'Yes'
            elif 'No' in response:
                answer = 'No'
            else:
                answer = None

            if answer:
                result = {"response": answer}
            else:
                result = {"response": "No"}
        else:
            result = {"error": "Unexpected output format from calculate_pipe."}
    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        result = {"error": str(e)}
    end = time.time()
    print(f"Time taken: {end - start} seconds, Response: {result}")
    return result


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, short_system=False):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        # confidence_1 = item['confidence_1']

        ## Prepare data for correct_sentences function
        sentence_data_1 = [
            {'sentence': ocr_text, 'file_name': item['file_name'], 'ground_truth_label': item['ground_truth_label']}]

        is_correct_1 = check_label_correctness(ocr_text, pipe)
        if "response" in is_correct_1 and is_correct_1["response"] == 'Yes':
            corrected_sentence_1 = ocr_text
            confidence_1 = "100"
            justification_1 = "There is no need to correct it, because it is already correct."
        else:
            corrected_sentence_1, confidence_1, justification_1 = correct_sentences_1(sentence_data_1, pipe, 10,
                                                                                      short_system)

        ## Compute CER for Mistral prediction
        cer_mistral_1 = cer_metric.compute(predictions=[corrected_sentence_1], references=[ground_truth_label])

        sentence_data_2 = [
            {'sentence': corrected_sentence_1, 'file_name': item['file_name'],
             'ground_truth_label': item['ground_truth_label']}]

        is_correct_2 = check_label_correctness(corrected_sentence_1, pipe)
        if "response" in is_correct_2 and is_correct_2["response"] == 'Yes':
            corrected_sentence_2 = corrected_sentence_1
            confidence_2 = "100"
            justification_2 = "There is no need to correct it, because it is already correct."
        else:
            corrected_sentence_2, confidence_2, justification_2 = correct_sentences_2(sentence_data_2, pipe, 10,
                                                                                      short_system)

        cer_mistral_2 = cer_metric.compute(predictions=[corrected_sentence_2], references=[ground_truth_label])

        # result = SelectBestSentence(ocr_text, corrected_sentence_1, pipe)
        # selected_label = result['selected_sentence']
        # cer_from_result = cer_metric.compute(predictions=[selected_label], references=[ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence_1': 0
            },
            'Check Label - Using OCR errors': {
                'is_correct_1': is_correct_1["response"],
                'predicted_label': corrected_sentence_1,
                'cer': cer_mistral_1,
                'confidence_1': int(confidence_1),
                'justification_1': justification_1
            },
            'Check Label - Using OCR examples': {
                'is_correct_2': is_correct_2["response"],
                'predicted_label': corrected_sentence_2,
                'cer': cer_mistral_2,
                'confidence_2': int(confidence_2),
                'justification_2': justification_2
            }
        })

    save_mistral_output = os.path.join(automated_resuts, name_file)
    save_to_json(results, save_mistral_output)


# Load Mistral model and tokenizer
## mistral_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Hypothetical correct model name
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_model = transformers.AutoModelForCausalLM.from_pretrained(mistral_model_name,
                                                                  torch_dtype=torch.float16,
                                                                  device_map="auto",
                                                                  token=TOKEN
                                                                  )
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=TOKEN, max_length=32)
mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, batch_size=10)

results_path_from_ocr = os.path.join(results_test_trocr, 'final_test_evaluation_results_25.json')
loaded_data = load_from_json(results_path_from_ocr)
## Example usage
evaluate_test_data_mistral7B(loaded_data, mistral_pipe, 'final_exp_8_v4.json', True)
print("The MISTRAL data is saved.")
