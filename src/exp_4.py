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


def check_sentence(sentence, context, pipe, short_system=False, ground_label=''):
    # if short_system:
    nummer_length = (len(sentence) * 2) + 1800
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
        # Post-processing to remove any additional unwanted text
        # response = response[1:-1]
        # response = response.replace("'", "")
        response = response.split('\n')[0].strip()

        # if (len(response)) > (len(sentence)*2):
        #     print(f"The response from MISTRAL is very long: {response}")
        #     response = sentence

        # if "[" in response:
        #     print(f"the response from MIXTRAL contains special characters: {response}")
        #     response = sentence

    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = sentence
    return response


def get_document_id(file_name):
    # Extract the first 3 digits of the file_name as the document ID
    return file_name.split("-")[0]


def correct_sentences(sentence_data, pipe, batch_size=10, short_system=False):
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

            processed_sentence = check_sentence(sentence, '', pipe, short_system, ground_truth_label)
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


def SelectBestSentence(ocr_sentence, mistral_sentence, pipe):
    nummer_length = ((len(ocr_sentence) + len(mistral_sentence)) * 2) + 500
    prompt = (f"<s>[INST] You are provided with two sentences: Both come from a Mistral model to correct OCR errors."
              f" Your task is to select the best Mistral sentence that corrects the OCR error and reduce the CER value:\n"
              f"1. The selected sentence.\n"
              f"2. The accuracy of the selected sentence in percentage.\n"
              f"3. The confidence in the selection in percentage.\n"
              f"4. A justification for the selection.\n"
              f"\nInputs:\n"
              f"    Mistral sentence 1: {ocr_sentence}\n"
              f"    Mistral sentence 2: {mistral_sentence}\n"
              f"\nExample:\n"
              f"    Mistral sentence 1: Hello htere\n"
              f"    Mistral sentence 2: Hello there\n"
              f"    Selected Sentence: Hello there\n"
              f"    Accuracy (%): 90\n"
              f"    Confidence (%): 95\n"
              f"    Justification: It accurately corrects the typo 'htere' to 'there', improving readability "
              f"and correctness.\n"
              f"\nLet's work this out in a step-by-step way to be sure we have the right answer."
              f"[/INST]</s>"
              )
    final_prompt = f"{prompt}\nThen the output is:"

    # print(f"Final Prompt: {final_prompt}")  # Debugging information

    try:
        corrected_text = calculate_pipe(pipe, final_prompt, nummer_length, 1)
        # print(f"Corrected Text: {corrected_text}")  # Debugging information

        if corrected_text and 'generated_text' in corrected_text[0]:
            response = corrected_text[0]['generated_text'].split('Then the output is:')[-1].strip()
            # print(f"Full Response: {response}")  # Debugging information

            # Extract the selected sentence, accuracy, and justification
            selected_sentence = ""
            accuracy = ""
            confidence = ""
            justification = ""

            lines = response.split('\n')
            for line in lines:
                if "Selected Sentence:" in line:
                    selected_sentence = line.split("Selected Sentence:")[-1].strip()
                elif "Accuracy (%):" in line:
                    accuracy = line.split("Accuracy (%):")[-1].strip()
                elif "Confidence (%):" in line:
                    confidence = line.split("Confidence (%):")[-1].strip()
                elif "Justification:" in line:
                    justification = line.split("Justification:")[-1].strip()

            # Ensure keys exist in the result dictionary
            result = {
                "selected_sentence": selected_sentence or "N/A",
                "accuracy": accuracy or "N/A",
                "confidence": confidence or "N/A",
                "justification": justification or "N/A"
            }
        else:
            result = {"error": "Unexpected output format from calculate_pipe."}

        # print(f"Result: {result}")  # Debugging information
    except Exception as e:
        print(f"Error in processing sentence '{ocr_sentence}': {e}")
        result = {"error": str(e)}

    return result


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, short_system=False):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        # confidence = item['confidence']

        ## Prepare data for correct_sentences function
        sentence_data = [
            {'sentence': ocr_text, 'file_name': item['file_name'], 'ground_truth_label': item['ground_truth_label']}]

        corrected_sentence_1, confidence_1, justification_1 = correct_sentences(sentence_data, pipe, 10, short_system)

        corrected_sentence_2, confidence_2, justification_2 = correct_sentences(sentence_data, pipe, 10, short_system)

        result = SelectBestSentence(corrected_sentence_1, corrected_sentence_2, pipe)

        ## Ensure mistral_text is a string

        ## mistral_text = corrected_sentences[0] if corrected_sentences else 'Error'

        ## Compute CER for Mistral prediction
        selected_label = result['selected_sentence']
        cer_from_result = cer_metric.compute(predictions=[selected_label], references=[ground_truth_label])

        # result = SelectBestSentence(ocr_text, corrected_sentence, pipe)
        # selected_label = result['selected_sentence']
        # cer_from_result = cer_metric.compute(predictions=[selected_label], references=[ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence': 0
            },
            'Select best Mistral label': {
                'predicted_label': selected_label,
                'accuracy': result['accuracy'],
                'confidence': result['confidence'],
                'cer': cer_from_result,
                'justification': result['justification']
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

results_path_from_ocr = os.path.join(results_test_trocr, 'testing.json')
loaded_data = load_from_json(results_path_from_ocr)
## Example usage
evaluate_test_data_mistral7B(loaded_data, mistral_pipe, 'final_exp_5.json', True)
print("The MISTRAL data is saved.")
