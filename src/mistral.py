import os
import time

import evaluate
import torch
import transformers
from transformers import AutoTokenizer, pipeline

from confidence_calculator import calculate_confidence
from handle_dataset import save_to_json, load_from_json
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


def check_sentence(sentence, context, pipe, short_system=False):
    if short_system:
        nummer_length = (len(sentence) * 2) + 1300
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
            "\n\n## Examples of Corrections:"
            "\n1. OCR Error from the User: '30th. Letters Orders and Instructions December 1755.' "
            "Correction from the assistant: '308th Letters, Orders, and Instructions, December 1755.'"
            "\n2. OCR Error from the User: 'remain here until the arrival of the usual with' "
            "Correction from the assistant: 'remain here until the arrival of the vessel with'"
            "\n3. OCR Error from the User: 'thes, Vc. and to me directions' "
            "Correction from the assistant: 'the Stores, &c., and to be under the same directions'"
            "\n4. OCR Error from the User: 'of Clothes; Shoes, Stockings, Shirts, Vc.: propatioma-' "
            "Correction from the assistant: 'of Clothes; Shoes, Stockings, Shirts, &c.: proportionate-'"
            "\n5. OCR Error from the User: 'things delivered into into the Nuggets, and are that' "
            "Correction from the assistant: 'Things delivered into the wagons, and see that'"
            "\n6. OCR Error from the User: 'No. To Doctor James Frank of the Virginia' "
            "Correction from the assistant: 'To Doctor James Craik, of the Virginia'"
            "\n7. OCR Error from the User: 'as before ordered. To soon as the Stores arrive, you' "
            "Correction from the assistant: 'as before ordered. As soon as the Stores arrive, you'"
            "\n8. OCR Error from the User: 'You are are immediately, upon receipt' "
            "Correction from the assistant: 'You are immediately, upon receipt'"
            "\n9. OCR Error from the User: 'hereof, to repair to Winchester, where you will never' "
            "Correction from the assistant: 'hereof, to repair to Winchester, where you will meet'"
            "\n10. OCR ERROR from the User: 'you that that, and processing, yourself to Winches-'"
            "Correction from the assistant: 'your Chest, and proceeding, yourself, to Winches-'"
            "</s>"
        )
    else:
        nummer_length = (len(sentence) * 2) + 2100
        system_prompt = (
            "<s>"
            "[INST] Task: Correct OCR errors found in a dataset of 18th-century documents. These errors range from simple "
            "misspellings to complex issues like incorrect abbreviations and term misinterpretations. Your objective is "
            "to correct these errors with precision, ensuring the integrity of the original manuscripts is maintained. "
            "Avoid introducing new information or omitting essential details Focus on retaining the original style, "
            "accuracy, and 18th-century linguistic conventions.[/INST]"
            "\n\n## Guidelines:"
            "\n- Address only OCR errors; please do not add more content."
            "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
            "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
            "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
            "\n\n## Examples of corrected sentences modelling cases:"
            "\n1. OCR Error from the User: '30th. Letters Orders and Instructions December 1755.' "
            "Correction from the assistant: '308th Letters, Orders, and Instructions, December 1755.'"
            "\n2. OCR Error from the User: 'remain here until the arrival of the usual with' "
            "Correction from the assistant: 'remain here until the arrival of the vessel with'"
            "\n3. OCR Error from the User: 'thes, Vc. and to me directions' "
            "Correction from the assistant: 'the Stores, &c., and to be under the same directions'"
            "\n4. OCR Error from the User: 'of Clothes; Shoes, Stockings, Shirts, Vc.: propatioma-' "
            "Correction from the assistant: 'of Clothes; Shoes, Stockings, Shirts, &c.: proportionate-'"
            "\n5. OCR Error from the User: 'things delivered into into the Nuggets, and are that' "
            "Correction from the assistant: 'Things delivered into the wagons, and see that'"
            "\n6. OCR Error from the User: 'No. To Doctor James Frank of the Virginia' "
            "Correction from the assistant: 'To Doctor James Craik, of the Virginia'"
            "\n7. OCR Error from the User: 'as before ordered. To soon as the Stores arrive, you' "
            "Correction from the assistant: 'as before ordered. As soon as the Stores arrive, you'"
            "\n8. OCR Error from the User: 'You are are immediately, upon receipt' "
            "Correction from the assistant: 'You are immediately, upon receipt'"
            "\n9. OCR Error from the User: 'hereof, to repair to Winchester, where you will never' "
            "Correction from the assistant: 'hereof, to repair to Winchester, where you will meet'"
            "\n10. OCR ERROR from the User: 'you that that, and processing, yourself to Winches-' "
            "Correction from the assistant: 'your Chest, and proceeding, yourself, to Winches-'"
            "\n11. OCR from the User: 'percent number of Waggons to carry them to Win-' "
            "Correction from the assistant: 'ficient number of waggons to carry them to Win-'"
            "\n12. OCR from the User: 'an Shirt Guard is kept our them, that no undes-' "
            "Correction from the assistant: 'a strict Guard is kept over them, that no embez-'"
            "\n13. OCR from the User: 'concerning Limited, Limited on their march upheros, there' "
            "Correction from the assistant: 'conveniently Quartered on their march up; there'"
            "\n14. OCR from the User: 'must be made of. If any of the Soldiers' "
            "Correction from the assistant: 'must be made use of. - If any of the Soldiers'"
            "\n15. OCR from the User: 'should direct front from this place, if would be proper' "
            "Correction from the assistant: 'should desert from this place, - it would be proper'"
            "\n16. PCR from the User: 'eds until for Duty, or Duty, on Review, are ordered to be' "
            "Correction from the assistant: 'ed unfit for Duty, on Review; are ordered to be'"
            "\n17. OCR from the User: 'up the Cobbs by the first pregnant' "
            "Correction from the assistant: 'up the Cask of Tools by the first waggon that'"
            "\n18. OCR from the User: 'the aid de camp. We Mr. Mr. We knowingly is to send' "
            "Correction from the assistant: 'the aid de camp. The Commissary is to send'"
            "\n19. OCR from the User: 'Colonel, Stephen will give you you an ac ac-' "
            "Correction from the assistant: 'Colonel Stephen will give you an ac-'"
            "\n20. OCR from the User: 'commissioned officered Officer or Solution should happen to' "
            "Correction from the assistant: 'commissioned Officer or Soldier should happen to'"
            "\n21. OCR from the User: 'You are to be particularly ex-' "
            "Correction from the assistant: 'You are to be particularly ex-'"
            "\n22. OCR from the User: 'that meeting with Letters at Fredericks-'"
            "Correction from the assistant: 'that meeting with Letters at Fredericks-'"
            "\n23. OCR from the User: 'proper Dispositions, and seeing that all our neces-' "
            "Correction from the assistant: 'proper Dispositions, and seeing that all our neces-'"
            "\n24 OCR from the User: 'have a very good effect. The Commonalty in gene-' "
            "Correction from the assistant: 'have a very good effect. The Commonalty in gene-'"
            "\n25. OCR from the User: 'money are greatest. When I left Willi-' "
            "Correction from the assistant: 'money are greatest. When I left Willi-'"
            "\n26. OCR from the User: 'with sundry arms, Vc. for the Troops. This Ves-' "
            "Correction from the assistant: 'with sundry arms, Vc. for the Troops. This Ves-'"
            "\n27. OCR from the User: 'me much concern; fearing that that this de-' "
            "Correction from the assistant: 'me much concern; fearing that this de-'"
            "\n28. OCR from the User: 'the Spring. If your Honor think proper to or-' "
            "Correction from the assistant: 'the Spring. If your Honor think proper to or-'"
            "\n29. OCR from the User: 'would fix it: so is Captain Stewarts: If Captain Stew-' "
            "Correction from the assistant: 'would fix it: so is Captain Stewarts: If Captain Stew-'"
            "\n30. OCR from the User: 'proper Dispositions, and seeing that all our neces-' "
            "Correction from the assistant: 'proper Dispositions, and seeing that all our neces-'"
            "\n31. OCR from the User: 'Letters Orders Orders and Instructions. Decembers. December 1755.' "
            "Correction from the assistant: 'Letters Orders and Instructions December. 1755.'"
            "\n32. OCR from the User: 'There importantly expecte' "
            "Correction from the assistant: 'I have impatiently expected'"
            "\n33. OCR from the User: 'the Ist. of December, to Rendezvous at Alexa-' "
            "Correction from the assistant: 'the 1st. of December, to Rendezvous at Alex-'"
            "</s>"
        )

    adaptation_request = (
        f"<s>"
        f"[INST] Based on the guidelines and illustrated examples, accurately correct the OCR errors in the following "
        f"sentence without adding extraneous information: '{sentence}'. If you don't know how to correct return the "
        f"original sentence as is: '{sentence}'.[/INST]"
        f"</s>"
    )

    prompt = f"{system_prompt}\n{adaptation_request}\nThe corrected sentence from the assistant:"
    # nummer_length = (len(sentence) * 2) + 2100

    try:
        corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
        response = corrected_text[0]['generated_text'].split('The corrected sentence from the assistant:')[-1].strip()
        # Post-processing to remove any additional unwanted text
        response = response[1:-1]
        response = response.replace("'", "")
        response = response.split('\n')[0].strip()

        if (len(response)) > (len(sentence)*2):
            print(f"The response from MISTRAL is very long: {response}")
            response = sentence

    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = sentence
    return response


def get_document_id(file_name):
    # Extract the first 3 digits of the file_name as the document ID
    return file_name.split("-")[0]


def correct_sentences(sentence_data, pipe, batch_size=10, short_system=False):
    global document_contexts
    # Ensure batch_size is never zero to prevent ValueError
    # if batch_size <= 0:
    #     print("Batch size must be greater than 0. Setting batch_size to 10.")
    #     batch_size = 10
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
                # print(GOOD_GRAMMAR)
            elif check_grammar_result == BAD_GRAMMAR:
                is_correct = BAD_GRAMMAR
                # print(BAD_GRAMMAR)
                corrected_sentence = check_sentence(sentence, context, pipe, short_system)
                sentence_to_append = corrected_sentence if corrected_sentence != 'Error' else sentence
            else:
                sentence_to_append = sentence
                print(f"Grammar check error or inconclusive for sentence: {sentence}")

            corrected_sentences.append(sentence_to_append)
            document_contexts[document_id].append(ground_truth_label)
            # print(f"Updated context for document {document_id}: {document_contexts[document_id]}")

    end = time.time()
    print(f"Time taken: {end - start} seconds, Processed {len(corrected_sentences)} sentences. "
          f"The sentence has a good grammar?: {is_correct}")
    return corrected_sentences


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, short_system=False):
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        confidence = item['confidence']

        # Prepare data for correct_sentences function
        sentence_data = [
            {'sentence': ocr_text, 'file_name': item['file_name'], 'ground_truth_label': item['ground_truth_label']}]

        corrected_sentences = correct_sentences(sentence_data, pipe, 10, short_system)

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

    save_mistral_output = os.path.join(automated_resuts, name_file)
    save_to_json(results, save_mistral_output)


# Load Mistral model and tokenizer
# mistral_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Hypothetical correct model name
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
# # Example usage
# evaluate_test_data_mistral7B(loaded_data, mistral_pipe, 'final.json', False)
# print("The MISTRAL data is saved.")

