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
from utils.constants import results_test_trocr, results_mixed_LLM_MISTRAL, automated_resuts, TOKEN, EXAMPLES, \
    EXAMPLES_F, EXAMPLES_F_60, OCR_ERRORS, EXAMPLES_F_3, EXAMPLES_F_4, EXAMPLES_V2_F_10, EXAMPLES_O_V2_10

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


def ocr_errors_corrector(sentence, pipe, mistral_tokenizer, reason=None):
    prompt = (
        f"You are an expert OCR (Optical Character Recognition) error checker. Your primary task is to meticulously "
        f"correct OCR errors, provide a justification, and indicate the percentage of confidence in your correction. "
        "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
        "linguistic conventions of the 18th century.The aim is to accurately reduce OCR errors in the text line. "
        f"\n\n## Guidelines: "
        f"\n- Address only OCR errors; do not add additional content. "
        f"\n- Ensure corrections accurately reflect 18th-century language, style, and conventions. \n"
        f"\n- Do not complete cut-off words at the end of the OCR text line. "
        f"\n- If the corrected text line is twice the length of the OCR text line, return the OCR text line. "
        f"\n- The confidence percentage reflects your certainty in the accuracy. "
        f"\n- Do not alter the ending of the text line. "
        f"\n\n## Illustrative example: "
        f"\n1. Given OCR text line: 'The quick brown fox jumpd over the lazy dog'. And given the explanation "
        f"by the user: 'The word 'jumpd' is an OCR error because it is a common' misspelling in scanned text. "
        f"The correct word in the context of 18th century is 'jumped'.'"
        f"\nThe JSON output is: "
        f"  Original text line: The quick brown fox jumpd over the lazy dog " 
        f"  Corrected text line: The quick brown fox jumped over the lazy dog "
        f"  Confidence (%): 98 "
        f"  Justification: Corrected 'jumpd' to 'jumped' based on the explanation. "
        f"\n[INST] Based on the guidelines, correct all the OCR errors of the text line: '{sentence}'. Use the explanation "
        f"given by the user for OCR errors: '{reason}'.[/INST]"
        f"\nThen, the JSON output is:"
    )

    start = time.time()
    try:
        nummer_length = count_tokens(prompt, mistral_tokenizer)
        number_token = nummer_length + 150
        corrected_text = calculate_pipe(pipe, prompt, number_token, 1)

        # Extract the raw generated text
        raw_response = corrected_text[0]['generated_text']
        # print(f"Raw response: {raw_response}")

        # Parse the response to extract the relevant parts
        json_output_marker = "Then, the JSON output is:"
        if json_output_marker in raw_response:
            json_output = raw_response.split(json_output_marker)[-1].strip()
            # original_sentence_match = re.search(r"Original sentence:\s*(.*?)(?:\s*Corrected sentence:|$)", json_output)
            corrected_sentence_match = re.search(r"Corrected text line:\s*(.*?)(?:\s*Confidence|$)", json_output)
            confidence_match = re.search(r"Confidence \(%\):\s*(\d+)", json_output)
            justification_match = re.search(r"Justification:\s*(.*)", json_output)

            # original_sentence = original_sentence_match.group(1).strip() if original_sentence_match else ""
            corrected_sentence = corrected_sentence_match.group(1).strip() if corrected_sentence_match else ""
            confidence = confidence_match.group(1).strip() if confidence_match else ""
            justification = justification_match.group(1).strip() if justification_match else ""

            if confidence == '':
                corrected_sentence = sentence
                confidence = 0
                justification = "An error has occurred."

            result = {
                # "Original sentence": original_sentence,
                "Corrected sentence": corrected_sentence,
                "Confidence (%)": confidence,
                "Justification": justification
            }

        else:
            result = {"error": "JSON output marker not found in the response"}
    except Exception as e:
        logger.error(f"Error in processing text line '{sentence}': {e}")
        result = {"error": str(e)}

    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, Response: {result}")
    return result


def check_ocr(sentence, pipe, mistral_token):
    prompt = (
        f"You will be provided with a text line, and you need to respond with 'Yes' or 'No' to the question: "
        f"Is the provided text line, that comes from a {MODEL}, Contains OCR (Optical Character Recognition) errors? "
        f"Additionally, you need to provide the explanation for your answer."
        f"\n\n## Illustrative examples:"
        f"{EXAMPLES_O_V2_10}"
        f"\n[INST] Now, please analyze the following text line and "
        f"provide your response along with the detailed explanation of all errors, "
        f"the text line contains OCR errors: '{sentence}' [/INST]\nThen the output is:"
    )
    start = time.time()
    try:
        nummer_length = count_tokens(prompt, mistral_token)
        number_token = nummer_length + GRACE_TOKEN
        corrected_text = calculate_pipe(pipe, prompt, number_token, 1)

        # Extract the raw generated text
        raw_response = corrected_text[0]['generated_text']
        # print(f"Raw response: {raw_response}")

        # Parse the response to extract the relevant parts
        json_output_marker = "Then the output is:"
        if json_output_marker in raw_response:
            json_output = raw_response.split(json_output_marker)[-1].strip()
            response_match = re.search(r"Response:\s*(Yes|No)", json_output)
            explanation_match = re.search(r"Explanation:\s*(.*)", json_output)

            response = response_match.group(1).strip() if response_match else ""
            explanation = explanation_match.group(1).strip() if explanation_match else ""

            result = {
                "response": response,
                "explanation": explanation
            }
        else:
            result = {"error": "JSON output marker not found in the response"}
    except Exception as e:
        logger.error(f"Error in processing text line '{sentence}': {e}")
        result = {"error": str(e)}

    end = time.time()
    logger.info(f"Time taken: {end - start} seconds, Response: {result}")
    return result


def evaluate_test_data_mistral7B(loaded_data, pipe, name_file, mistral_tokenizer):
    results = []
    for item in loaded_data:
        ocr_text = item['predicted_label']
        ground_truth_label = item['ground_truth_label']
        is_correct = check_ocr(ocr_text, pipe, mistral_tokenizer)

        if "response" in is_correct and 'Yes' in is_correct["response"]:
            # reason = is_correct.get("reason", "No reason provided.")
            explanation = is_correct.get("explanation", "No explanation provided.")
            # sentence_data = [{'sentence': ocr_text}]
            result_from_corrector = ocr_errors_corrector(
                ocr_text,
                pipe,
                mistral_tokenizer,
                explanation
            )
            corrected_sentence = result_from_corrector["Corrected sentence"]
            confidence = result_from_corrector["Confidence (%)"]
            justification = result_from_corrector["Justification"]
            # cer_mistral = cer_metric.compute(predictions=[corrected_sentence], references=[ground_truth_label])
            cer_mistral = evaluation.cer_only([corrected_sentence], [ground_truth_label])
        else:
            # reason = is_correct["reason"]
            explanation = is_correct["explanation"]
            corrected_sentence = ocr_text
            confidence = "100"
            justification = "There is no need to correct it, because it is already correct."
            cer_mistral = item['cer']


        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': ground_truth_label,
            'OCR': {
                'predicted_label': ocr_text,
                'cer': item['cer'],
                'confidence': 0
            },
            'Prompt checking': {
                'response': is_correct["response"],
                # 'reason': reason,
                'explanation': explanation
            },
            'Prompt correcting': {
                'predicted_label': corrected_sentence,
                'cer': cer_mistral,
                'confidence': int(confidence),
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
