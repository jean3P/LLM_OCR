import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from jiwer import cer

from src.handle_dataset_washington import save_to_json, load_from_json
from src.utils.constants import automated_resuts, results_test_trocr


def load_model_and_tokenizer(model_dir="/home/pool/fine_tunning_llm/Fine_tunning/results_v6_100000"):
    """
    Load the base model and tokenizer, and apply the fine-tuned LoRA adapters.

    Parameters:
        model_dir (str): The directory containing the fine-tuned model checkpoint.

    Returns:
        model: The fine-tuned model with LoRA adapters.
        tokenizer: The tokenizer for the model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Ensure all GPUs are visible
    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Load the model with LoRA configuration
    try:
        model = PeftModel.from_pretrained(base_model, model_dir)
    except TypeError as e:
        print(f"Error loading LoRA model: {e}")
        raise

    return model, tokenizer


def correct_ocr_output(ocr_output_text, model, tokenizer):
    """
    Use the fine-tuned model to correct OCR output text.

    Parameters:
        ocr_output_text (str): The OCR output text to be corrected.
        model: The fine-tuned model with LoRA adapters.
        tokenizer: The tokenizer for the model.

    Returns:
        corrected_text (str): The corrected text.
    """
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
        "\n\n# Examples of corrected sentences modelling cases"
        "\n1. Input: 2oh. Leters Orders and Instructions December 175. "
        "     Corrected label: 308th Letters, Orders, and Instructions, December 1755."
        "\n2. Input: remain here until the arival of the vesel with "
        "     Corrected label: remain here until the arrival of the vessel with"
        "\n3. Input: as befou ordered. So son as the Stores arive, you "
        "     Corrected label: as before ordered. So soon as the Stores arrive, you"
        "\n4. Input: are, with al posible dispatch, te procure a suf- "
        "     Corrected label: are, with all possible dispath, to procure a suf-"
        "\n5. Input: ficient number of Wagons to cary them to Fn- "
        "     Corrected label: ficient number of waggons to carry them to Win-"
        "\n6. Input: thester; whither they are to be sent, ander the "
        "     Corrected label: chester; whither they are to be sent, under the"
        "\n7. Input: escort of the Soldiers now here, except the Suits "
        "     Corrected label: escort of the Soldiers now here: except the Suits"
        "\n8. Input: of Clothes; Shoes, Stockings, Shirts, Vc. proportiona- "
        "     Corrected label: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona-"
        "\n9. Input: bly, which are to be lft with botonel Carby le. "
        "      Corrected label: bly which are to be left with Colonel Carlyle."
        "\n10. Input: Alexandria: December 1th. 175. "
        "      Corrected label: Alexandria: December 16th. 1755."
        "\n11. Input: sent to Staford, to him there. "
        "      Corrected label: sent to Stafford, to him there."
        "\n12. Input: ately of the Recruits now in this tomn, by the sweral "
        "      Corrected label: ately of the Recruits now in this town, by the several."
        "\n13. Input: Oficers who enlisted them; mentioning their height, "
        "      Corrected label: Officers who enlisted them; mentioning their height,"
        "\n14. Input: age, trade, Vc. The Oficers to se that the Serge- "
        "      Corrected label: age, trade, Vc. The officers to see that the Serge-"
        "\n15. Input: Rp.30g. "
        "      Corrected label: p.309."
        "\n16. Input: I amVc. "
        "      Corrected label: I am Vc."
        "\n17.  Input: G.W. Aid tecamp. "
        "      Corrected label: G.W. aid de camp."
        "\n18. Input: 28th. Parole Albemarle WinchesterD December 15. 275. "
        "      Corrected label: 20th. Parole Abbemarle. Winchester: December 20th. 1755."
        "\n19. Input: dis char ged: Vir3. "
        "      Corrected label: discharged: viz."
        "</s>"
    )
    adaptation_request = (
        f"<s>"
        f"[INST] Based on the guidelines and illustrated examples, accurately correct the OCR errors in the following "
        f"sentence.[/INST]</s>"
        f"\n\n### Input:\n{ocr_output_text}\n\n### Corrected label:"
    )
    eval_prompt = (
        f"{system_prompt}\n{adaptation_request}"
    )
    model_input = tokenizer(eval_prompt, return_tensors="pt", padding=True).to("cuda")

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=100,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id to avoid warnings
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)
    start = generated_text.find("### Corrected label:") + len("### Corrected label:")
    end = generated_text.find("### Input:", start)
    if end == -1:
        end = None  # In case it's the last section
    corrected_text = generated_text[start:end].strip().split('\n')[0]

    if (len(corrected_text)) > (len(ocr_output_text) * 1.5):
        # print("HERE!!!!")
        corrected_text = ocr_output_text

    return corrected_text


def evaluate_model(test_file, model, tokenizer, output_file="evaluation_results.json"):
    """
    Evaluate the fine-tuned model on a test dataset and save results.

    Parameters:
        test_file (str): The path to the test dataset file.
        model: The fine-tuned model with LoRA adapters.
        tokenizer: The tokenizer for the model.
        output_file (str): The path to save the evaluation results.
    """
    # with open(test_file, "r", encoding="utf-8") as f:
    #     test_data = json.load(f)

    results = []
    for item in test_file:
        true_label = item["ground_truth_label"]
        ocr_label = item["predicted_label"]

        start_time = time.time()
        corrected_label = correct_ocr_output(ocr_label, model, tokenizer)
        cer_value = cer(true_label, corrected_label)
        end_time = time.time()
        time_taken = end_time - start_time
        results.append({
            'file_name': item['file_name'],
            'ground_truth_label': true_label,
            'OCR': {
                'predicted_label': ocr_label,
                'cer': item['cer'],
                'confidence': 0
            },
            'MISTRAL': {
                'predicted_label': corrected_label,
                'cer': cer_value
            }
        })
        print(f"Corrected OCR in {time_taken:.2f} seconds,\n "
              f"    OCR Label: {ocr_label} | Corrected Label: {corrected_label}")
    save_mistral_output = os.path.join(automated_resuts, output_file)
    save_to_json(results, save_mistral_output)


# if __name__ == "__main__":
#     # Example usage
#     test_file = os.path.join(results_test_trocr, "final_test_evaluation_results_25.json")
#     load_test = load_from_json(test_file)
#     model, tokenizer = load_model_and_tokenizer()
#     evaluate_model(load_test, model, tokenizer)
#     print("Evaluation completed and results saved.")
