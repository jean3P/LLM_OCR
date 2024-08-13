import json
import os
import random
from collections import defaultdict

from src.utils.constants import results_test_trocr


def compare_labels(ground_truth, predicted):
    """Compare two strings and return sets of unique errors."""
    substitution_errors = defaultdict(set)
    insertion_errors = set()
    deletion_errors = set()

    len_gt = len(ground_truth)
    len_pred = len(predicted)
    max_len = max(len_gt, len_pred)

    for i in range(max_len):
        if i >= len_gt:
            insertion_errors.add(f"Insertion: '{predicted[i]}'")
        elif i >= len_pred:
            deletion_errors.add(f"Deletion: '{ground_truth[i]}'")
        elif ground_truth[i] != predicted[i]:
            substitution_errors[ground_truth[i]].add(predicted[i])

    return substitution_errors, insertion_errors, deletion_errors


def analyze_ocr_errors(json_files):
    """Analyze OCR errors from multiple JSON files."""
    substitution_errors_set = defaultdict(set)
    insertion_errors_set = set()
    deletion_errors_set = set()

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_name = os.path.basename(json_file)  # Extract filename from path

        print(f"Total entries in {json_file}: {len(data)}")

        for entry in data:
            if 'ground_truth_label' not in entry or 'predicted_label' not in entry:
                print(f"Entry missing required fields in {file_name}: {entry}")
                continue

            ground_truth = entry['ground_truth_label']
            predicted = entry['predicted_label']

            substitution_errors, insertion_errors, deletion_errors = compare_labels(ground_truth, predicted)

            for key, value_set in substitution_errors.items():
                substitution_errors_set[key].update(value_set)

            insertion_errors_set.update(insertion_errors)
            deletion_errors_set.update(deletion_errors)

    return substitution_errors_set, insertion_errors_set, deletion_errors_set


def print_errors(substitution_errors_set, insertion_errors_set, deletion_errors_set):
    """Print errors in a readable format."""
    print("Substitution Errors:")
    for key, value_set in substitution_errors_set.items():
        print(f"- '{key}' -> {value_set}")

    print("\nInsertion Errors:")
    for error in insertion_errors_set:
        print(f"- {error}")

    print("\nDeletion Errors:")
    for error in deletion_errors_set:
        print(f"- {error}")


def generate_mistral_7b_response(substitution_errors_set, insertion_errors_set, deletion_errors_set):
    """Generate Mistral-7b response prompt for correcting OCR errors."""
    response = "Mistral-7b Prompt:\n"
    response += "Please correct the following OCR errors:\n\n"

    response += "Substitution Errors:\n"
    for key, value_set in substitution_errors_set.items():
        response += f"- '{key}' -> {list(value_set)}\n"

    response += "\nInsertion Errors:\n"
    for error in insertion_errors_set:
        response += f"- {error}\n"

    response += "\nDeletion Errors:\n"
    for error in deletion_errors_set:
        response += f"- {error}\n"

    return response


def apply_ocr_errors(true_labels, substitution_errors_set, insertion_errors_set, deletion_errors_set):
    """Apply OCR errors to the true labels."""
    ocr_examples_with_errors = []

    for example in true_labels:
        ground_truth = example['ground_truth_label']
        predicted = list(ground_truth)  # Convert to list to allow modifications
        len_gt = len(ground_truth)
        num_errors = max(1, len_gt * 25 // 100)  # Apply a 30% error rate

        error_positions = random.sample(range(len_gt), num_errors)

        # Apply substitution errors
        for pos in error_positions:
            char = ground_truth[pos]
            if char in substitution_errors_set and substitution_errors_set[char]:
                predicted[pos] = random.choice(list(substitution_errors_set[char]))

        # Randomly choose between insertion and deletion for the remaining errors
        remaining_positions = list(set(range(len_gt)) - set(error_positions))
        remaining_positions = random.sample(remaining_positions, max(0, num_errors - len(error_positions)))

        for pos in remaining_positions:
            if random.choice([True, False]) and pos < len(predicted):
                # Apply insertion error
                if insertion_errors_set:
                    predicted.insert(pos, random.choice(list(insertion_errors_set)).split(": '")[1].strip("'"))
            elif pos < len(predicted):
                # Apply deletion error
                del predicted[pos]

        ocr_examples_with_errors.append({
            "ground_truth_label": ground_truth,
            "predicted_label": ''.join(predicted)
        })

    return ocr_examples_with_errors


# Example usage:
if __name__ == "__main__":
    json_files = [
        'final_test_evaluation_results_100.json',
        'final_test_evaluation_results_75.json',
        'final_test_evaluation_results_50.json',
        'final_test_evaluation_results_25.json'
    ]
    # List of JSON files to analyze; adjust filenames as necessary

    full_paths = [os.path.join(results_test_trocr, json_file) for json_file in json_files]

    try:
        # Analyze OCR errors from JSON files
        substitution_errors_set, insertion_errors_set, deletion_errors_set = analyze_ocr_errors(full_paths)

        # Print the detailed errors
        print_errors(substitution_errors_set, insertion_errors_set, deletion_errors_set)

        # True labels for OCR examples
        # ocr_examples = [
        #     {"ground_truth_label": "only for the publick use, unless by particu-"},
        #     {"ground_truth_label": "down a Barrel of Flints with the Arms, to"},
        #     {"ground_truth_label": "Plantation of Charles Sellars - the rest to Captain"},
        #     {"ground_truth_label": "The Officers who came down"},
        #     {"ground_truth_label": "to Repair to Captain Hoggs Company with eight"},
        #     {"ground_truth_label": "one Sergeant, one Corporal, one Drummer,"},
        #     {"ground_truth_label": "without a Surgeon, if you will do that duty, an"},
        #     {"ground_truth_label": "halt there until he joins, in order to escort the"},
        #     {"ground_truth_label": "are to supply him from the"},
        #     {"ground_truth_label": "march them immediately to Fort Cumberland;"},
        #     {"ground_truth_label": "Winchester, and about two thousand weight"},
        #     {"ground_truth_label": "Captain Ashby and Company, at the"},
        #     {"ground_truth_label": "28th Winchester: October 28th, 1755."},
        #     {"ground_truth_label": "The Officers who came down"},
        #     {"ground_truth_label": "28th. To Ensign Fleming, of the Virginia Regiment."},
        #     {"ground_truth_label": "Captain Hogg's Company at Fort Dinwiddie"},
        #     {"ground_truth_label": "I received yours of the 6th. of Octo-"},
        #     {"ground_truth_label": "Winchester, and about two thousand weight"},
        #     {"ground_truth_label": "Recruiting; and they are allowed until the 1st. of De-"},
        #     {"ground_truth_label": "each man distinguishes"},
        #     {"ground_truth_label": "Captain Ashby and Company, at the"},
        #     {"ground_truth_label": "Plantation of Charles Sellars - the rest to Captain"},
        #     {"ground_truth_label": "The Commissary is to see"},
        #     {"ground_truth_label": "If any have received Sergeants pay, it"},
        #     {"ground_truth_label": "When you arrive at Winchester, you must provide your men with Car-"}
        # ]
        # ocr_examples = [
        #     {"ground_truth_label": "only for the publick use, unless by particu-"},
        #     {"ground_truth_label": "down a Barrel of Flints with the Arms, to"},
        #     {"ground_truth_label": "Plantation of Charles Sellars - the rest to Captain"},
        #     {"ground_truth_label": "The Officers who came down"},
        #     {"ground_truth_label": "to Repair to Captain Hoggs Company with eight"},
        #     {"ground_truth_label": "one Sergeant, one Corporal, one Drummer,"},
        #     {"ground_truth_label": "without a Surgeon, if you will do that duty, an"},
        #     {"ground_truth_label": "halt there until he joins, in order to escort the"},
        #     {"ground_truth_label": "are to supply him from the"},
        #     {"ground_truth_label": "march them immediately to Fort Cumberland;"},
        #     {"ground_truth_label": "Winchester, and about two thousand weight"},
        #     {"ground_truth_label": "Captain Ashby and Company, at the"},
        #     {"ground_truth_label": "28th Winchester: October 28th, 1755."},
        #     {"ground_truth_label": "Parole Hampton."},
        #     {"ground_truth_label": "from Fort Cumberland with Colonel"},
        #     {"ground_truth_label": "Washington, are immediately to go Recrui-"},
        #     {"ground_truth_label": "ting; and they are allowed until the 1st. of De-"},
        #     {"ground_truth_label": "cember; at which time if they do not"},
        #     {"ground_truth_label": "punctually appear at the place of Rendez-"},
        #     {"ground_truth_label": "vous assigned them, they will be tried by a"},
        #     {"ground_truth_label": "Court Martial, for disobedience of Orders."},
        #     {"ground_truth_label": "They are to wait upon the Aid de camp"},
        #     {"ground_truth_label": "at one of the Clock, to receive their Recrui-"},
        #     {"ground_truth_label": "ting Instructions. Each Officer present, to give"},
        #     {"ground_truth_label": "in a Return immediately of the number"},
        #     {"ground_truth_label": "of men he has enlisted. - One Subaltern,"},
        #     {"ground_truth_label": "one Sergeant, one Corporal, one Drummer,"},
        #     {"ground_truth_label": "and twenty five private men, are to mount"},
        #     {"ground_truth_label": "Guard to - day, and to be relieved to - morrow"},
        #     {"ground_truth_label": "at ten o'clock. -. All Reports and Returns"},
        #     {"ground_truth_label": "are to be made to the Aid de Camp."},
        #     {"ground_truth_label": "28th. To Ensign Fleming, of the Virginia Regiment."},
        #     {"ground_truth_label": "You are hereby ordered to repair to"},
        #     {"ground_truth_label": "Captain Hogg's Company at Fort Dinwiddie"},
        #     {"ground_truth_label": "with eight good men: as that Company is"},
        #     {"ground_truth_label": "without a Surgeon, if you will do that duty, an"},
        #     {"ground_truth_label": "allowance will be made you for it. You are"},
        #     {"ground_truth_label": "to provide medicines, Vc. upon the best terms"},
        #     {"ground_truth_label": "you can. This Order I expect will be immedi-"},
        #     {"ground_truth_label": "ately complied with; and that no Delays be of-"},
        #     {"ground_truth_label": "fered. You are to account with Captain Bell"},
        #     {"ground_truth_label": "for your recruiting money, before you leave him."},
        #     {"ground_truth_label": "If you should arrive at Augusta Court House"},
        #     {"ground_truth_label": "before Sergeant Wilper and his Party, you are to"},
        #     {"ground_truth_label": "halt there until he joins, in order to escort the"},
        #     {"ground_truth_label": "Ammunition, Vc. for the Fort; where you will"},
        #     {"ground_truth_label": "receive Clothes and Arms for the Men."},
        #     {"ground_truth_label": "28th. To Captain Bell of the Virginia Regiment."},
        #     {"ground_truth_label": "I have Ordered Ensign Fleming"},
        #     {"ground_truth_label": "to Repair to Captain Hoggs Company with eight"}
        # ]

        # ocr_examples = [
        #     {"ground_truth_label": "Dear Sir, Oxon, Sept. 17, 1734."},
        #     {"ground_truth_label": "I Had the favour of your letter last Friday, which brought"},
        #     {"ground_truth_label": "me the agreeable news of your and Mrs, H,'s welfare, to-"},
        #     {"ground_truth_label": "gether with the much-desired account of your approving the"},
        #     {"ground_truth_label": "scheme, inclosed in my last. Indeed, I did not doubt of its"},
        #     {"ground_truth_label": "meeting; with a candid reception, from all those persons to"},
        #     {"ground_truth_label": "whom it was recommended. Their known concern for reli-"},
        #     {"ground_truth_label": "gion, giving me sufficient assurance, that nothing can be un-"},
        #     {"ground_truth_label": "acceptable to them, which any way tended to promote their"},
        #     {"ground_truth_label": "improvement in the divine life. It must be confessed, indeed,"},
        #     {"ground_truth_label": "as you very justly observed, that we must make a great pro-"}
        # ]

        ocr_examples = [
            {"ground_truth_label": "Dear Sir, Oxon, Sept. 17, 1734."},
            {"ground_truth_label": "I Had the favour of your letter last Friday, which brought"},
            {"ground_truth_label": "me the agreeable news of your and Mrs, H,'s welfare, to-"},
            {"ground_truth_label": "gether with the much-desired account of your approving the"},
            {"ground_truth_label": "scheme, inclosed in my last. Indeed, I did not doubt of its"},
            {"ground_truth_label": "meeting; with a candid reception, from all those persons to"},
            {"ground_truth_label": "whom it was recommended. Their known concern for reli-"},
            {"ground_truth_label": "gion, giving me sufficient assurance, that nothing can be un-"},
            {"ground_truth_label": "acceptable to them, which any way tended to promote their"},
            {"ground_truth_label": "improvement in the divine life. It must be confessed, indeed,"},
            {"ground_truth_label": "as you very justly observed, that we must make a great pro-"},
            {"ground_truth_label": "gress in religion, and be inured by frequent prayer and medi-"},
            {"ground_truth_label": "tation, to the extatic contemplation of heavenly objects, before"},
            {"ground_truth_label": "we can arrive at true heavenly-mindedness; and, perhaps, after"},
            {"ground_truth_label": "all our endeavours, whilst our souls are immersed in these"},
            {"ground_truth_label": "fleshly tabernacles, we shall make but very small advances in so"},
            {"ground_truth_label": "delightful and glorious an undertaking. But believe me, Sir,"},
            {"ground_truth_label": "you cannot imagine, how vastly serviceable the constant use of"},
            {"ground_truth_label": "all the means of religion will be, in acquiring this blessed habit"},
            {"ground_truth_label": "of mind. Such, as an early rising in the morning, public"},
            {"ground_truth_label": "and private prayer, a due temperance in all things, and fre-"},
            {"ground_truth_label": "quent meditation on the infinite love and purity of that un-"},
            {"ground_truth_label": "paralleled pattern of all perfection, our dear Redeemer. As"},
            {"ground_truth_label": "for your mentioning, Sir, the degeneracy of the age, as the"},
            {"ground_truth_label": "least objection against our making further advances in any"},
            {"ground_truth_label": "religious improvement, I cannot by any means admit of it."}
        ]

        # Apply OCR errors to the true labels
        ocr_examples_with_errors = apply_ocr_errors(ocr_examples, substitution_errors_set, insertion_errors_set,
                                                    deletion_errors_set)

        # Print the ground truth and OCR errors
        for example in ocr_examples_with_errors:
            print(f"Ground truth: {example['ground_truth_label']}")
            print(f"OCR with errors: {example['predicted_label']}\n")

        # Generate and print the Mistral-7b response
        mistral_7b_response = generate_mistral_7b_response(substitution_errors_set, insertion_errors_set,
                                                           deletion_errors_set)
        print(mistral_7b_response)
    except Exception as e:
        print(f"Error analyzing files: {e}")
