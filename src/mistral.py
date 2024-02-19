import os
import time

import evaluate
import torch
import transformers
from transformers import AutoTokenizer, pipeline

from confidence_calculator import calculate_confidence
from handle_dataset import save_to_json, load_from_json
from utils.constants import results_test_trocr, WASHINGTON_SAMPLE_TEXT, results_LLM_mistral_2

BAD_GRAMMAR = 'No'

GOOD_GRAMMAR = 'Yes'

device = torch.device("cpu")
cer_metric = evaluate.load('cer')

# Global variable to store context sentences for each document
document_contexts = {}


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


def check_sentence(sentence, context, pipe):
    system_prompt = (
        f"<s>Your task is to adjust the language of transcriptions of historical documents, that were scanning "
        f"using OCR technics, to reflect the original 18th-century style. Specifically, you are working "
        f"with the George Washington dataset (here there is a sample of the dataset: {WASHINGTON_SAMPLE_TEXT}). "
        f"Hint: The corrected preserves the same number of words of the sentence from the OCR."
        f"## Examples:"
        f"  Sentence from OCR:'and the houses given to him to him. You meet' - the corrected sentence: "
        f"'and the horses given to him. You must' "
        f"  Sentence from OCR:'The Bauer, Captain, John Mercer.' "
        f"- the corrected sentence: 'The Bearer, captain John Mercer,'"
        f"  Sentence from OCR:'itaughetti. Pay make my make my Components' "
        f"- the corrected sentence: 'slaughter. Pray make my Compliments'"
        f"  Sentence from OCR:'8rd.30. To Sergeant David Wilper, of the Virginia' "
        f"- the corrected sentence:'Octo. 30th. To Sergeant David Wilper, of the Virginia'"
        f"  Sentence from OCR:'t. To Mr. Boyd, Paymaster' - the corrected sentence: '1st. To Mr. Boyd, Paymaster.' "
        f"  Sentence from OCR: 'During here, no provision is to be de-' "
        f"- the corrected sentence: 'During his stay here, no provision is to be de-' "
        f"  Sentence from OCR: 'Every twenty Blancly Blanks.' - the corrected sentence: 'liver twenty Blankets.'"
        f"  Sentence from OCR: '6:30th. To Sergeant David Wilper, of the Virginia' "
        f"- the corrected sentence: 'Octo. 30th. To Sergeant David Wilper, of the Virginia' "
        f"  Sentence from OCR: 'perhaps near the third when none but stronger' "
        f"- the corrected sentence: 'perhaps near the Ohio) when none but strong par-' "
        f"  Sentence from OCR: 'your, and perceiving to Winches-' "
        f"- the corrected sentence: 'your Chest, and proceeding, yourself, to Winches-'</s>")

    if context:  # when there are previously corrected sentences
        adaptation_request = (
            f"<s>[INST] Given the context of the sentence: {context}, please correct any "
            f"errors in the following sentence from OCR: '{sentence}' to match the 18th-century language "
            f"style, without adding any extra information.[/INST]</s>")
    else:  # when there are no previously corrected sentences
        adaptation_request = (
            f"<s>[INST] Without any prior context, correct the sentence from OCR: '{sentence}' to reflect "
            f"the language style of the 18th century, without adding extra information.[/INST]</s>")

    prompt = f"{system_prompt}\n{adaptation_request}\nThe corrected sentence is:"
    nummer_length = (len(sentence) * 2) + 1000

    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        response = corrected_text[0]['generated_text'].split('The corrected sentence is:')[-1].strip()
        # Post-processing to remove any additional unwanted text
        response = response[1:-1]
        response = response.replace("'", "")
        response = response.split('\n')[0].strip()

    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = 'Error'
    return response


def standardize_terms(sentence, context, pipe):
    # Assuming check_standardize_terms is similar to check_spelling, returning 'Yes' if standardization is needed
    standardization_result = check_standardize_terms(sentence, pipe)
    print(f"The sentence has terms that require standardization: {standardization_result}")

    if standardization_result == 'Yes':
        system_prompt = (
            "<s>Identify and standardize only the terms and abbreviations in the provided sentence, "
            "originating from an OCR process, to ensure they reflect accurate historical terminology and usage. "
            "Preserve the original sentence structure and punctuation, including dashes. "
            "## Examples: "
            "OCR sentence: 'Refer to the doc. as per the inst. given.' "
            "- Standardized sentence: 'Refer to the document as per the instructions given.'"
            "OCR sentence: 'Arrival of the Vc. with the supplies.' - "
            "Standardized sentence: 'Arrival of the viz. with the supplies.'</s>"
        )

        if context:  # When there are previously corrected sentences to provide context
            adaptation_request = (
                f"<s>[INST] Given the previous sentences of the document that help understand the context: {context}, "
                f"please standardize the terms and abbreviations of the OCR sentence: '{sentence}' [/INST]</s>")
        else:  # When there is no prior context available
            adaptation_request = (
                f"<s>[INST] Without any prior context, standardize the terms and abbreviations of the OCR sentence: '{sentence}' "
                f"[/INST]</s>")

        prompt = f"{system_prompt}\n{adaptation_request}\nStandardized sentence:"
        nummer_length = (len(sentence) * 2) + 500

        try:
            corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
            response = corrected_text[0]['generated_text'].split('Standardized sentence:')[-1].strip()
            # Post-processing to remove any additional unwanted text
            response = response[1:-1]
            response = response.replace("'", "")
            response = response.split('\n')[0].strip()

        except Exception as e:
            print(f"Error in processing sentence '{sentence}': {e}")
            response = 'Error'
        return response

    elif standardization_result == 'No':
        print("No terms require standardization:", sentence)
        return sentence

    else:
        print("Error or ambiguity detected in term standardization check.")
        return sentence


def check_spelling_mistakes(sentence, context, pipe):
    spelling_result = check_spelling(sentence, pipe)
    print(f"Te sentence has spelling mistakes: {spelling_result}")
    if spelling_result == 'Yes':
        system_prompt = (
            f"<s>Identify and correct only the spelling mistakes in the provided sentence, "
            f"originating from an OCR process, to accurately reflect the original text in 18th-century style. "
            f"Ensure the corrected sentence retains the same number of words as the OCR version. "
            f"## Examples:"
            f"OCR sentence: '30th. Letters Orders and Instructions December 1755.' - Corrected sentence: 'the Stores'"
            f"OCR sentence: 'thes, Vc. and to me directions' "
            f"- Corrected sentence: 'the Stores, viz. and to be under the same directions'"
            f"</s>")

        if context:  # when there are previously corrected sentences
            adaptation_request = (
                f"<s>[INST] Given the previous sentences of the document that help understand the context: {context}, "
                f"please correct the spelling mistakes of the OCR sentence: '{sentence}' [/INST]</s>")
        else:  # when there are no previously corrected sentences
            adaptation_request = (
                f"<s>[INST] Without any prior context, correct the spelling mistakes of OCR sentence: '{sentence}' "
                f"[/INST]</s>")

        prompt = f"{system_prompt}\n{adaptation_request}\nCorrected sentence:"
        nummer_length = (len(sentence) * 2) + 500

        try:
            corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
            response = corrected_text[0]['generated_text'].split('Corrected sentence:')[-1].strip()
            # Post-processing to remove any additional unwanted text
            response = response[1:-1]
            response = response.replace("'", "")
            response = response.split('\n')[0].strip()

        except Exception as e:
            print(f"Error in processing sentence '{sentence}': {e}")
            response = 'Error'
        return response
    elif spelling_result == 'No':
        print("No spelling mistakes found:", sentence)
        return sentence
    else:
        print("Error or ambiguity detected in spelling check.")
        return sentence


def get_document_id(file_name):
    # Extract the first 3 digits of the file_name as the document ID
    return file_name.split("-")[0]


def correct_sentences(sentence_data, pipe, batch_size=10):
    global document_contexts
    start = time.time()
    corrected_sentences = []
    is_correct = ''
    for i in range(0, len(sentence_data), batch_size):
        batch = sentence_data[i:i + batch_size]
        for data in batch:
            sentence = data['sentence']
            file_name = data['file_name']
            ground_truth_label = data['ground_truth_label']
            document_id = get_document_id(file_name)

            if document_id not in document_contexts:
                document_contexts[document_id] = []

            context = "\n".join(document_contexts[document_id])
            # print(f"Context before processing for document {document_id}: {context}")

            check_grammar_result = check_grammar(sentence, pipe)
            if check_grammar_result == GOOD_GRAMMAR:
                sentence_to_append = sentence
                is_correct = GOOD_GRAMMAR
                print(GOOD_GRAMMAR)
            elif check_grammar_result == BAD_GRAMMAR:
                is_correct = BAD_GRAMMAR
                print(BAD_GRAMMAR)
                # context = context + "\n ---Here the sentence from the OCR---"
                corrected_sentence = check_spelling_mistakes(sentence, context, pipe)
                corrected_sentence = standardize_terms(corrected_sentence, context, pipe)
                sentence_to_append = corrected_sentence if corrected_sentence != 'Error' else sentence
            else:
                sentence_to_append = sentence
                # print(f"Grammar check error or inconclusive for sentence: {sentence}")

            corrected_sentences.append(sentence_to_append)
            document_contexts[document_id].append(ground_truth_label)
            # print(f"Updated context for document {document_id}: {document_contexts[document_id]}")

    end = time.time()
    print(f"Time taken: {end - start} seconds, Processed {len(corrected_sentences)} sentences. "
          f"The sentence has a good grammar?: {is_correct}")
    return corrected_sentences


def evaluate_test_data(loaded_data, pipe):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        confidence = item['confidence']

        # Prepare data for correct_sentences function
        sentence_data = [
            {'sentence': ocr_text, 'file_name': item['file_name'], 'ground_truth_label': item['ground_truth_label']}]

        corrected_sentences = correct_sentences(sentence_data, pipe)

        # Ensure mistral_text is a string
        mistral_text = corrected_sentences[0] if corrected_sentences else 'Error'

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

    save_mistral_output = os.path.join(results_LLM_mistral_2, 'evaluation_results_with_mistral_1.json')
    save_to_json(results, save_mistral_output)


# Load Mistral model and tokenizer
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_model = transformers.AutoModelForCausalLM.from_pretrained(mistral_model_name, torch_dtype=torch.float16,
                                                                  device_map="auto")
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, batch_size=10)

results_path_from_ocr = os.path.join(results_test_trocr, 'test_evaluation_results_seq_mini_2.json')
loaded_data = load_from_json(results_path_from_ocr)
# Example usage
evaluate_test_data(loaded_data, mistral_pipe)
print("The MISTRAL data is saved.")
