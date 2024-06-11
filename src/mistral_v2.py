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

BAD_GRAMMAR = 'No'

GOOD_GRAMMAR = 'Yes'

device = torch.device("cuda")
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


def check_sentence(sentence, context, pipe, short_system=False, ground_label=''):
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
    # else:
    #     nummer_length = (len(sentence) * 2) + 1600
    #     system_prompt = (
    #         "<s>"
    #         "[INST] Your primary task is to meticulously correct OCR (Optical Character Recognition) errors in a "
    #         "collection of 18th-century documents. These documents contain a variety of errors, ranging from simple "
    #         "misspellings to more complex issues like incorrect abbreviations and misinterpretations of terms. "
    #         "Your corrections must strive for precision, preserving the authenticity and integrity of the original "
    #         "manuscripts. It's imperative to avoid introducing new information or excluding essential details. "
    #         "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
    #         "linguistic conventions of the 18th century.[/INST]"
    #         "\n\n## Guidelines:"
    #         "\n- Address only OCR errors; please do not add more content."
    #         "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
    #         "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
    #         "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
    #         "\n\n## Examples of corrected sentences modelling cases:"
    #         "\n1. OCR Error from the User: 30th. Letters Orders and Instructions December 1755. "
    #         "   The JSON output is:"
    #         "     Corrected sentence: 308th Letters, Orders, and Instructions, December 1755."
    #         "     Percentage of confidence: 99"
    #         "     Justification: The number '30th' is corrected to '308th' as it seems to be a mistake in the original text. The addition of commas and capitalization of 'Orders' and 'Instructions' improves the sentence structure and readability."
    #         "\n2. OCR Error from the User: remain here until the arrival of the usual with"
    #         "   The JSON output is:"
    #         "     Corrected sentence: remain here until the arrival of the vessel with"
    #         "     Percentage of confidence: 95"
    #         "     Justification: The word 'thes' is replaced with 'the Stores' as it seems to be a contextually appropriate correction. The addition of 'to be' after '&c.' improves the sentence structure."
    #         "\n3. OCR Error from the User: you that that, and processing, yourself to Winches-"
    #         "   The JSON output is:"
    #         "     Corrected sentence: your Chest, and proceeding, yourself, to Winches-"
    #         "     Percentage of confidence: 80"
    #         "     Justification: The repetition of 'that' is removed and 'your Chest' is added as it seems to be a contextually appropriate correction."
    #         "\n4 OCR Error from the User: thes, Vc. and to me directions"
    #         "   The JSON output is:"
    #         "     Corrected sentence: the Stores, &c., and to be under the same directions"
    #         "     Percentage of confidence: 90"
    #         "     Justification: The word 'thes' is replaced with 'the Stores' as it seems to be a contextually appropriate correction. The addition of 'to be' after '&c.' improves the sentence structure."
    #         "\n5 OCR Error from the User: of Clothes; Shoes, Stockings, Shirts, Vc.: propatioma-"
    #         "   The JSON output is:"
    #         "     Corrected sentence: of Clothes; Shoes, Stockings, Shirts, &c.: proportionate-"
    #         "     Percentage of confidence: 85"
    #         "     Justification: The word 'Vc.' is replaced with '&c.' as it seems to be a contextually appropriate correction. The addition of 'proportionate' after '&c.' improves the sentence structure."
    #         "\n6 OCR Error from the User: things delivered into into the Nuggets, and are that"
    #         "   The JSON output is:"
    #         "     Corrected sentence: Things delivered into the wagons, and see that"
    #         "     Percentage of confidence: 80"
    #         "     Justification: The word 'Nuggets' is replaced with 'wagons' as it seems to be a contextually appropriate correction. The removal of one 'into' and the addition of 'see' improves the sentence structure."
    #         "\n7 OCR Error from the User: No. To Doctor James Frank of the Virginia"
    #         "   The JSON output is:"
    #         "     Corrected sentence: To Doctor James Craik, of the Virginia"
    #         "     Percentage of confidence: 99"
    #         "     Justification: The word 'Frank' is replaced with 'Craik' as it seems to be a contextually appropriate correction. The addition of 'To' at the beginning and 'of the' after 'Craik' improves the sentence structure."
    #         "\n8 OCR Error from the User: as before ordered. To soon as the Stores arrive, you"
    #         "   The JSON output is:"
    #         "     Corrected sentence: as before ordered. As soon as the Stores arrive, you"
    #         "     Percentage of confidence: 95"
    #         "     Justification: The word 'To' is replaced with 'As' as it seems to be a contextually appropriate correction. The addition of 'As' before 'soon' improves the sentence structure."
    #         "\n9 OCR Error from the User: You are are immediately, upon receipt"
    #         "   The JSON output is:"
    #         "     Corrected sentence: You are immediately, upon receipt"
    #         "     Percentage of confidence: 90"
    #         "     Justification: The repetition of 'are' is removed as it seems to be a mistake in the original text."
    #         "\n10 OCR Error from the User: hereof, to repair to Winchester, where you will never"
    #         "   The JSON output is:"
    #         "     Corrected sentence: hereof, to repair to Winchester, where you will meet"
    #         "     Percentage of confidence: 85"
    #         "     Justification: The word 'never' is replaced with 'meet' as it seems to be a contextually appropriate correction."
    #         "\n11 OCR Error from the User: percent number of Waggons to carry them to Win-"
    #         "   The JSON output is:"
    #         "     Corrected sentence: fficient number of waggons to carry them to Win-"
    #         "     Percentage of confidence: 80"
    #         "     Justification: The word 'percent' is replaced with 'fficient' as it seems to be a contextually appropriate correction."
    #         "\n12 OCR Error from the User: an Shirt Guard is kept our them, that no undes-"
    #         "   The JSON output is:"
    #         "     Corrected sentence: a strict Guard is kept over them, that no embez-"
    #         "     Percentage of confidence: 85"
    #         "     Justification: The words 'an Shirt' are replaced with 'a strict' and 'our' is replaced with 'over' as they seem to be contextually appropriate corrections."
    #         "\n13 OCR Error from the User: concerning Limited, Limited on their march upheros, there"
    #         "   The JSON output is:"
    #         "     Corrected sentence: conveniently Quartered on their march up; there"
    #         "     Percentage of confidence: 80"
    #         "     Justification: The words 'concerning Limited, Limited' are replaced with 'conveniently Quartered' as they seem to be contextually appropriate corrections."
    #         "</s>"
    #     )

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


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, short_system=False):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        # confidence = item['confidence']

        ## Prepare data for correct_sentences function
        sentence_data = [
            {'sentence': ocr_text, 'file_name': item['file_name'], 'ground_truth_label': item['ground_truth_label']}]

        corrected_sentence, confidence, justification = correct_sentences(sentence_data, pipe, 10, short_system)

        ## Ensure mistral_text is a string
        ## mistral_text = corrected_sentences[0] if corrected_sentences else 'Error'

        ## Compute CER for Mistral prediction
        cer_mistral = cer_metric.compute(predictions=[corrected_sentence], references=[ground_truth_label])

        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence': 0
            },
            'MISTRAL': {
                'predicted_label': corrected_sentence,
                'cer': cer_mistral,
                'confidence': int(confidence),
                'justification': justification
            }
        })

    save_mistral_output = os.path.join(automated_resuts, name_file)
    save_to_json(results, save_mistral_output)


## Load Mistral model and tokenizer
# ## mistral_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Hypothetical correct model name
# mistral_model_name = "mistralai/Mistral-7B-v0.1"
# mistral_model = transformers.AutoModelForCausalLM.from_pretrained(mistral_model_name,
#                                                                   torch_dtype=torch.float16,
#                                                                   device_map="auto",
#                                                                   token=TOKEN
#                                                                   )
# mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=TOKEN, max_length=32)
# mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, batch_size=10)
#
# results_path_from_ocr = os.path.join(results_test_trocr, 'testing.json')
# loaded_data = load_from_json(results_path_from_ocr)
# ## Example usage
# evaluate_test_data_mistral7B(loaded_data, mistral_pipe, 'final_3.json', True)
# print("The MISTRAL data is saved.")

