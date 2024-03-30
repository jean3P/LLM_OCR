import time
import evaluate
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from confidence_calculator import calculate_confidence
import torch
from handle_dataset import save_to_json, load_from_json
from utils.constants import automated_resuts, results_test_trocr

cer_metric = evaluate.load('cer')

BAD_GRAMMAR = 'No'

GOOD_GRAMMAR = 'Yes'


def calculate_response(messages, model, tokenizer):

    with torch.no_grad():
        token_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            temperature=0.5,
            do_sample=True,
            top_p=0.95,
            top_k=1,
            max_new_tokens=35,
        )
    output = tokenizer.decode(output_ids[0][token_ids.size(1):])
    return output


def check_grammar(ocr_text, model, tokenizer):
    messages = [
        {"role": "user", "content": f"Answer Yes or No: Is this sentence grammatically correct? '{ocr_text}'"}
    ]
    try:
        output = calculate_response(messages, model, tokenizer)
        if 'Yes' in output:
            grammar_result = 'Yes'
        elif 'No' in output:
            grammar_result = 'No'
        else:
            grammar_result = 'Error'
    except Exception as e:
        print(f"Error in processing sentence '{ocr_text}': {e}")
        grammar_result = 'Error'
    return grammar_result


def generate_correction(ocr_text, short_system, model, tokenizer):
    if short_system:
        messages = [
            {"role": "user",
             "content": "As an expert in 18th-century manuscripts, your task is to meticulously correct OCR "
                        "(Optical Character Recognition) errors in a collection of historical documents from the 1700s. "
                        "These documents contain various errors, including misspellings, incorrect abbreviations, "
                        "and misinterpreted terms."
                        "\n\n## When making corrections, adhere to the following guidelines::"
                        "\n- Address only OCR errors; please do not add more content."
                        "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
                        "\n- If words are cut off at the end of the OCR text or at the beginning, do not try to complete them."
                        "\n- If the corrected text would be more than twice the length of the original, return the original OCR text unmodified. "
                        "\n\n## Here are some examples of how to properly correct OCR errors::"
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
                        "Correction from the assistant: 'your Chest, and proceeding, yourself, to Winches-' "
                        "\n11. OCR ERROR from the User: 'fcient numbers to carry them to Win-' "
                        "Correction from the assistant: 'ficient number of waggons to carry them to Win-' "
                        f"Based on the guidelines and illustrated examples, accurately correct the OCR "
                        f"errors in the following sentence without adding extraneous information: "
                        f"'{ocr_text}'. If you are unsure how to correct any errors, "
                        f"return the original OCR text unmodified. with this format: "
                        f"\nThe corrected sentence is:"},
        ]
    else:
        messages = [
            {"role": "user",
             "content": "As an expert in 18th-century manuscripts, your task is to meticulously correct OCR "
                        "(Optical Character Recognition) errors in a collection of historical documents from the 1700s. "
                        "These documents contain various errors, including misspellings, incorrect abbreviations, "
                        "and misinterpreted terms."
                        "\n\n## When making corrections, adhere to the following guidelines::"
                        "\n- Address only OCR errors; please do not add more content."
                        "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
                        "\n- If words are cut off at the end of the OCR text or at the beginning, do not try to complete them."
                        "\n- If the corrected text would be more than twice the length of the original, return the original OCR text unmodified. "
                        "\n\n## Here are some examples of how to properly correct OCR errors:"
                        "\n1. OCR from the User: 'You are to be particularly ex-' "
                        "Correction from the assistant: 'You are to be particularly ex-'"
                        "\n2. OCR from the User: 'that meeting with Letters at Fredericks-'"
                        "Correction from the assistant: 'that meeting with Letters at Fredericks-'"
                        "\n3. OCR from the User: 'proper Dispositions, and seeing that all our neces-' "
                        "Correction from the assistant: 'proper Dispositions, and seeing that all our neces-'"
                        "\n4 OCR from the User: 'have a very good effect. The Commonalty in gene-' "
                        "Correction from the assistant: 'have a very good effect. The Commonalty in gene-'"
                        "\n5. OCR from the User: 'money are greatest. When I left Willi-' "
                        "Correction from the assistant: 'money are greatest. When I left Willi-'"
                        "\n6. OCR from the User: 'with sundry arms, Vc. for the Troops. This Ves-' "
                        "Correction from the assistant: 'with sundry arms, Vc. for the Troops. This Ves-'"
                        "\n7. OCR from the User: 'me much concern; fearing that that this de-' "
                        "Correction from the assistant: 'me much concern; fearing that this de-'"
                        "\n8. OCR from the User: 'the Spring. If your Honor think proper to or-' "
                        "Correction from the assistant: 'the Spring. If your Honor think proper to or-'"
                        "\n9. OCR from the User: 'would fix it: so is Captain Stewarts: If Captain Stew-' "
                        "Correction from the assistant: 'would fix it: so is Captain Stewarts: If Captain Stew-'"
                        "\n10. OCR from the User: 'proper Dispositions, and seeing that all our neces-' "
                        "Correction from the assistant: 'proper Dispositions, and seeing that all our neces-'"
                        "\n11. OCR from the User: 'Letters Orders Orders and Instructions. Decembers. December 1755.' "
                        "Correction from the assistant: 'Letters Orders and Instructions December. 1755.'"
                        "\n12. OCR from the User: 'There importantly expecte' "
                        "Correction from the assistant: 'I have impatiently expected'"
                        "\n13. OCR from the User: 'the Ist. of December, to Rendezvous at Alexa-' "
                        "Correction from the assistant: 'the 1st. of December, to Rendezvous at Alex-' "
                        f"Based on the guidelines and illustrated examples, accurately correct the OCR "
                        f"errors in the following sentence without adding extraneous information: "
                        f"'{ocr_text}'. If you are unsure how to correct any errors, "
                        f"return the original OCR text unmodified. with this format: "
                        f"\nThe corrected sentence is:"},
        ]

    output = calculate_response(messages, model, tokenizer)
    # Extract the sentence immediately following "The corrected sentence is: '"
    # print(output)
    start_phrase = "The corrected sentence is: '"
    end_phrase = "'"

    start_idx = output.find(start_phrase) + len(start_phrase)
    if start_idx > -1:
        # Find the ending quote that marks the end of the corrected sentence
        end_idx = output.find(end_phrase, start_idx)
        if end_idx > -1:
            # Extract and return the corrected sentence without the surrounding quotes
            response = output[start_idx:end_idx]
        else:
            response = "Corrected sentence not found. (End marker missing)"
    else:
        response = "Corrected sentence not found. (Start marker missing)"

    if (len(response)) > (len(ocr_text) * 1.5):
        print(f"the response from MIXTRAL is very long: {response}")
        response = ocr_text
    if "[" in response or "(" in response or "\n" in response:
        print(f"the response from MIXTRAL contains special characters: {response}")
        response = ocr_text
    return response


def correct_sentences(sentence_data, short_system, model, tokenizer):
    batch_size = 5
    global document_contexts
    start = time.time()
    corrected_sentences = []
    is_correct = ''
    for i in range(0, len(sentence_data), batch_size):
        batch = sentence_data[i:i + batch_size]
        for data in batch:
            sentence = data['sentence']

            check_grammar_result = check_grammar(sentence, model, tokenizer)
            if check_grammar_result == GOOD_GRAMMAR:
                sentence_to_append = sentence
                is_correct = GOOD_GRAMMAR
            elif check_grammar_result == BAD_GRAMMAR:
                is_correct = BAD_GRAMMAR
                corrected_sentence = generate_correction(sentence, short_system, model, tokenizer)
                sentence_to_append = corrected_sentence if corrected_sentence != 'Error' else sentence
            else:
                sentence_to_append = sentence
                print(f"Grammar check error or inconclusive for sentence: {sentence}")

            corrected_sentences.append(sentence_to_append)
    end = time.time()
    print(f"Time taken: {end - start} seconds, Processed {len(corrected_sentences)} sentences. "
          f"The sentence has a good grammar?: {is_correct}")
    return corrected_sentences


def evaluate_test_data_mixtral7B(loaded_data, name_file, short_system):
    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=False,
    )
    results = []

    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        confidence = item['confidence']

        sentence_data = [
            {'sentence': ocr_text, 'file_name': item['file_name'], 'ground_truth_label': item['ground_truth_label']}]

        corrected_sentences = correct_sentences(sentence_data, short_system, model, tokenizer)

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

    del model
    del tokenizer

# #
# #
# results_path_from_ocr = os.path.join(results_test_trocr, 'final_test_evaluation_results_75_25.json')
# loaded_data = load_from_json(results_path_from_ocr)
# # Example usage
# evaluate_test_data_mixtral7B(loaded_data, 'final_2.json', False)
# print("The MISTRAL data is saved.")
