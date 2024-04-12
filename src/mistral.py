import os
import time

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


def check_sentence(sentence, context, pipe, short_system=False):
    if short_system:
        nummer_length = (len(sentence) * 2) + 2200
        system_prompt = (
            "<s>"
            "[INST] Your primary task is to meticulously correct OCR (Optical Character Recognition) errors in a "
            "collection of 20th-century IAM documents. These documents contain a variety of errors, ranging from simple"
            " misspellings to more complex issues like incorrect abbreviations and misinterpretations of terms. "
            "Your corrections must strive for precision, preserving the authenticity and integrity of the original "
            "manuscripts. It's imperative to avoid introducing new information or excluding essential details. "
            "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
            "linguistic conventions of the 20th-century.[/INST]"
            "\n\n## Guidelines:"
            "\n- Address only OCR errors; please do not add more content at the end of the sentence."
            "\n- Ensure corrections accurately reflect the 20th-century language, style, and conventions."
            "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
            "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
            "\n\n## Examples of Corrections:"
            "\n1. OCR from the User: 'guard this as a satining-up gross and' "
            "Correction from the assistant: 'Regard this as a softening-up process, and'"
            "\n2. OCR from the User: 'have dinner with me. On the Lordship,' "
            "Correction from the assistant: 'have dinner with me? On His Lordship,'"
            "\n3. OCR from the User: 'HE SPANCED OFFIDONNUM, \"I ARE THE WAR. SEE BEEN GENTS' "
            "Correction from the assistant: 'he shrugged diffidently, \"I like the work. One gets plenty'"
            "\n4. OCR from the User: '\"Robbi\" and friend; what? You knew that' "
            "Correction from the assistant: '\"Robbie\" and friend', what? You know the'"
            "\n5. OCR from the User: 'sat of thing. \"You'll get nothing out of a-ne,\"' "
            "Correction from the assistant: 'sort of thing. \"\" You'll get nothing out of me, \"'"
            "\n6. OCR from the User: '\'Good deal. One for the grill-room destiny, and' "
            "Correction from the assistant: '\" Good-oh! One for the grill-room, darling, and'"
            "\n7. OCR from the User: 'John aged him stealthy. \"It's the Purcil' "
            "Correction from the assistant: 'John eyed him steadily.\" It's the Parcifal'"
            "\n8. OCR from the User: 'the Henry - why should I pick on Earlier? \'' "
            "Correction from the assistant: 'the Navy - why should I pick on Pericles? \"'"
            "\n9. OCR from the User: 'anything - knock twice and ask for Bouley, what? \'' "
            "Correction from the assistant: 'any time - knock twice and ask for Bawley, what? \"'"
            "\n10. OCR from the User: 'silences, for she knew they beckoned a story of' "
            "Correction from the assistant: 'silences, for she knew they betokened a worry of'"
            "\n11. OCR from the User: 'ALSO HOPERNY.' "
            "Correction from the assistant: 'asked hopefully.'"
            "\n12. OCR from the User: '\"BRONMAN! I HAVE LIKE IS SUTATIVES GREDELL, GRAYN. \"AM, INCW,' "
            "Correction from the assistant: '\"Bless you! I hope Luke is suitably grateful, darling.\" \"Oh, well,\"'"
            "\n13. OCR from the User: 'Bend. Supplies and of the duck. Blanked' "
            "Correction from the assistant: 'Dead. Shuffled out of the deck. Blasted'"
            "\n14. OCR from the User: 'mining camps, \"he said ablyuply. He checked' "
            "Correction from the assistant: 'mining camp,\" he said abruptly. He wheeled'"
            "\n15. OCR from the User: 'with screws, drinking a big men by a' "
            "Correction from the assistant: 'into draws, circling a big mesa by a'"
            "\n16. OCR from the User: 'four or the mile values, sandy-flavored. It' "
            "Correction from the assistant: 'four or five mile valley, sandy-floored. It'"
            "\n17. OCR from the User: 'your advisers off:\" \"Table it long, winter,\" I'sang' "
            "Correction from the assistant: 'your whiskers off. \"\" Take it easy, mister, \" I sang'"
            "\n18. OCR from the User: 'Moralle then, loud and grill. The gun borel' "
            "Correction from the assistant: 'whistle then, loud and shrill. The gun barrel'"
            "\n19. OCR from the User: 'came up close, every us surprisingly. \"We've' "
            "Correction from the assistant: 'came up close, eyeing us suspiciously.\" We've'"
            "\n20. OCR from the User: 'Golightly, Swede and the Sodbuster. This last was' "
            "Correction from the assistant: 'beautifully, Sweden and Southwark. This last was'"
            "\n21. OCR from the User: 'edging for gold but were bases in arms' "
            "Correction from the assistant: 'digging for gold but were babes in arms'"
            "\n22. OCR from the User: 'the didn't know what to make of it, but it' "
            "Correction from the assistant: 'He didn't know what to make of it, but it'"
            "\n23. OCR from the User: 'was Mr. Conference all right. Perhaps he had' "
            "Correction from the assistant: 'was Mr. Copthorne all right. Perhaps he had'"
            "\n24. OCR from the User: 'been att ending some kind of fancy dress do' "
            "Correction from the assistant: 'been attending some kind of fancy dress do'"
            "\n25. OCR from the User: 'in London. Very strange his eyes looked. Mud,' "
            "Correction from the assistant: 'in London. Very strange his eyes looked. Mad,'"
            "\n26. OCR from the User: 'hundred eyes, as if the poor fellow had works in' "
            "Correction from the assistant: 'hunted eyes, as if the poor fellow had rocks in'"
            "\n27. OCR from the User: 'vehicle as as' "
            "Correction from the assistant: '-----------------------------------------------------'"
            "\n28. OCR from the User: '\"You've very early about, sir,\" he said, hoping to' "
            "Correction from the assistant: '\"You're very early about, sir,\" he said, trying to'"
            "\n29. OCR from the User: 'illness. I travelled down from London' "
            "Correction from the assistant: '\"Yes. I travelled down from London'"
            "\n30. OCR from the User: 'the terrace thinking that perhaps Gashi' "
            "Correction from the assistant: 'the terrace thinking that perhaps Gavin'"
            "\n31. OCR from the User: 'the looked round morning the talks,' "
            "Correction from the assistant: 'dly decadence and absorption in debase-'"
            "\n32. OCR from the User: 'ally dependence and consumption in debate -' "
            "Correction from the assistant: 'Gavin had come back to her.'"
            "\n33. OCR from the User: 'Because she was second best. The others came back to' "
            "Correction from the assistant: 'Because she was second best. The others came back'"
            "\n34. OCR from the User: 'The table, all of then making afus of her and at once she' "
            "Correction from the assistant: 'to the table, all of them making a fuss of her and at once'"
            "\n35. OCR from the User: '\"What about it, my sweet?\" Well, \"she laughed' "
            "Correction from the assistant: '\"What about it, my sweet?\" \"Well,\" she laughed'"
            "\n36. OCR from the User: 'replied glass and looked over the rim at his' "
            "Correction from the assistant: 'refilled glass and looked over the rim at his'"
            "\n37. OCR from the User: 'guard this as a satining-up gross and' "
            "Correction from the assistant: 'Regard this as a softening-up process, and'"
            "\n38. OCR from the User: 'the Henry - why should I pick on Earlier? \'' "
            "Correction from the assistant: 'the Navy - why should I pick on Pericles? \"'"
            "\n39. OCR from the User: 'John aged him stealthy. \"It's the Purcil' "
            "Correction from the assistant: 'John eyed him steadily.\" It's the Parcifal'"
            "\n40. OCR from the User: 'Moralle then, loud and grill. The gun borel' "
            "Correction from the assistant: 'whistle then, loud and shrill. The gun barrel'"
            "\n41. OCR from the User: 'you could see they were hard it. They were' "
            "Correction from the assistant: 'you could see they were hard hit. They were'"
            # "\n42. OCR from the User: '' "
            # "Correction from the assistant: ''"
            "</s>"
        )
    else:
        nummer_length = (len(sentence) * 2) + 2299
        system_prompt = (
            "<s>"
            "[INST] Your primary task is to meticulously correct OCR (Optical Character Recognition) errors in a "
            "collection of 20th-century IAM documents. These documents contain a variety of errors, ranging from simple"
            " misspellings to more complex issues like incorrect abbreviations and misinterpretations of terms. "
            "Your corrections must strive for precision, preserving the authenticity and integrity of the original "
            "manuscripts. It's imperative to avoid introducing new information or excluding essential details. "
            "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
            "linguistic conventions of the 20th-century.[/INST]"
            "\n\n## Guidelines:"
            "\n- Address only OCR errors; please do not add more content at the end of the sentence."
            "\n- Ensure corrections accurately reflect the 20th-century language, style, and conventions."
            "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
            "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
            "\n\n## Examples of corrected sentences modelling cases:"
            "\n1. OCR from the User: 'Celestiallites using the position' "
            "Correction from the assistant: 'Gaitskellites using the position'"
            "\n2. OCR from the User: 'would formally to overturn them her' "
            "Correction from the assistant: 'around frantically to overturn them next'"
            "\n3. OCR from the User: 'as an taxes - use decisionly carried.' "
            "Correction from the assistant: 'as on bases - were decisively carried.'"
            "\n4. OCR from the User: 'This, and the number of active' "
            "Correction from the assistant: 'This, and the numbers of active'"
            "\n5. OCR from the User: 'being a low unto themselves and appealing' "
            "Correction from the assistant: 'being a law unto themselves and opposing'"
            "\n6. OCR from the User: 'later, but are in favour of German lover and' "
            "Correction from the assistant: 'bases, but are in favour of German bases and'"
            "\n7. OCR from the User: 'Congress, approaching B.W. All organized' "
            "Correction from the assistant: 'Congress, representing 800, 000 organised'"
            "\n8. OCR from the User: 'Trade unionists, when it decided by overwhelming' "
            "Correction from the assistant: 'trade unionists, when it decided by overwhelming'"
            "\n9. OCR from the User: 'than a handful of new operations' "
            "Correction from the assistant: 'than a handful of new apprenticeships'"
            "\n10. OCR from the User: 'angels Antonio's \"La Note\" The same*' "
            "Correction from the assistant: 'angelo Antonioni's \"La Notte.\" The num-'"
            "\n11. OCR from the User: 'regularity and author, wrote: \"Which's Wachs' "
            "Correction from the assistant: 'playwright and author, wrote:\" Wicki's blacks'"
            "\n12. OCR from the User: 'with Dutch bounty, building up to a' "
            "Correction from the assistant: 'with brutal honesty, building up to a'"
            "\n13. OCR from the User: 'the placing location of a Dyakhouse.' "
            "Correction from the assistant: 'the glaring beacon of a lighthouse.'"
            "\n14. OCR from the User: 'masculine jobs; more has learned anything. Nick:' "
            "Correction from the assistant: 'marvellous joke; no one has learned anything. Wicki'"
            "\n15. OCR from the User: 'succeed succeeded in his second film in' "
            "Correction from the assistant: 'succeeded in his second film in'"
            "\n16. PCR from the User: '\"Letairliche Godichken \"(180), five start stories' "
            "Correction from the assistant: '\"Unheimliche Geschichten\" (1920) , five ghost stories'"
            "\n17. OCR from the User: 'to the frame out its edges produced off' "
            "Correction from the assistant: 'to the frame and its edges planed off'"
            "\n18. OCR from the User: 'to the frame and its edges planed off' "
            "Correction from the assistant: 'piece of timber 14 in. long and about 1/2 in.'"
            "\n19. OCR from the User: 'risks cut in the moulds and into the extentes' "
            "Correction from the assistant: 'slots cut in the moulds and into the notches'"
            "\n20. OCR from the User: 'Korean Fatting, Glue, screw and make fast.' "
            "Correction from the assistant: 'transom framing. Glue, screw and make fast.'"
            "\n21. OCR from the User: 'the 5/8 by No. 1s gauge seems (born) or small' "
            "Correction from the assistant: 'Use 5/8 by No. 4 gauge screws (brass) or nail'"
            "\n22. OCR from the User: 'with inch and a greater palamized nails, if'"
            "Correction from the assistant: 'with inch and a quarter galvanized nails, if'"
            "\n23. OCR from the User: 'and you were teacher. You will be in' "
            "Correction from the assistant: 'and your own teacher. You will then be'"
            "\n24 OCR from the User: 'Poly-Poyco (fed and a member of the' "
            "Correction from the assistant: 'Rolls-Royce Ltd. and a member of the'"
            "\n25. OCR from the User: 'The Road Panest, Brewing studied the report on \"The House and' "
            "Correction from the assistant: 'The Lord Provost, having studied the report on'The Morale and'"
            "\n26. OCR from the User: 'its applications and he wonder the automatically of' "
            "Correction from the assistant: 'its implications and to consider the advisability of'"
            "\n27. OCR from the User: 'crafts, talks and believe time intervals' "
            "Correction from the assistant: 'crafts, hobbies and leisure-time interests'"
            "\n28. OCR from the User: 'god honors. Under and Chelsea! I'm afraid' "
            "Correction from the assistant: 'good humour. Ladies and gentlemen! I'm afraid'"
            "\n29. OCR from the User: 'was found at Wakebler. Neither the states not the Barry Maxwell it' "
            "Correction from the assistant: 'was found at Waterloo. Neither the sisters nor Mr Berry thought it'"
            "\n30. OCR from the User: 'likely that she had been carrying much money. It prefer at Houmlaw' "
            "Correction from the assistant: 'likely that she had been carrying much money. A porter at Hounslow'"
            "\n31. OCR from the User: 'impmESTies in gluom, and the crash of revered' "
            "Correction from the assistant: 'immersion in gloom, and the clash of reverberated'"
            "\n32. OCR from the User: 'The Mensheimer Tunnel, on the London London, so-' "
            "Correction from the assistant: 'The Merstham Tunnel, on the London-to-'"
            "\n33. OCR from the User: 'who was worked as a bookeszer.' "
            "Correction from the assistant: 'whom she worked as a book-keeper.'"
            "\n34. OCR from the User: 'He was there automatically. Dressed' "
            "Correction from the assistant: 'He was there punctually. Dressed'"
            "\n35. OCR from the User: 'I could meet you at the bus stain.\"' "
            "Correction from the assistant: 'I could meet you at the bus station. \"'"
            "\n36. OCR from the User: 'top of it. \"No,\" he said at last.\" I saw it allow that' "
            "Correction from the assistant: 'top of it. \"No,\" he said at last. \"I can't allow'"
            "\n37. OCR from the User: 'arrival of the lead and wire. When the penny was even, he' "
            "Correction from the assistant: 'arrival of the food and wine. When the pouring ritual was over he'"
            "\n38. OCR from the User: 'Hugh wassi impressed with this imaginative bar-' "
            "Correction from the assistant: 'Hugh wasn't impressed with this ingratiating bar-'"
            "\n39. OCR from the User: 'Lyle, Bobby reopened quickly. And once there he was' "
            "Correction from the assistant: 'Lydd', Roddy interposed quickly. 'And once there he was'"
            "\n40. OCR from the User: 'Council he met someone, either of Charlie or' "
            "Correction from the assistant: 'Obviously he met someone, either by chance or'"
            "\n41. OCR from the User: 'Hugh nodded. His face were the intention to appear' "
            "Correction from the assistant: 'Hugh nodded. His face wore the melancholy expression'"
            "\n42. OCR from the User: 'hereby first that it does high to find on the visit he rated' "
            "Correction from the assistant: 'Roddy puffed at his pipe, his eyes fixed on the white-coated'"
            "\n43. OCR from the User: 'Fred picked up about Pete Party always teaching' "
            "Correction from the assistant: 'Fred picked up about Pete Jesty always touching'"
            "\n44. OCR from the User: 'one thing for it, then, 'he said, visiting on' "
            "Correction from the assistant: 'one thing for it, Mum,\" he said, sitting on'"
            "\n45. OCR from the User: 'magazine was plumb and plain with a glassizing' "
            "Correction from the assistant: 'Maggie was plump and plain with a pleasing'"
            "\n46. OCR from the User: 'minutes involving and here directed at' "
            "Correction from the assistant: 'minable howling and were directed at'"
            # "\n47. OCR from the User: '' "
            # "Correction from the assistant: ''"
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

        if (len(response)) > (len(sentence)*1.5):
            print(f"The response from MISTRAL is very long: {response}")
            response = sentence

        if "[" in response:
            print(f"the response from MIXTRAL contains special characters: {response}")
            response = sentence

    except Exception as e:
        print(f"Error in processing sentence '{sentence}': {e}")
        response = sentence
    return response

# def check_sentence(sentence, context, pipe, short_system=False):
#     if short_system:
#         nummer_length = (len(sentence) * 2) + 1300
#         system_prompt = (
#             "<s>"
#             "[INST] Your primary task is to meticulously correct OCR (Optical Character Recognition) errors in a "
#             "collection of 18th-century documents. These documents contain a variety of errors, ranging from simple "
#             "misspellings to more complex issues like incorrect abbreviations and misinterpretations of terms. "
#             "Your corrections must strive for precision, preserving the authenticity and integrity of the original "
#             "manuscripts. It's imperative to avoid introducing new information or excluding essential details. "
#             "Your focus should be on maintaining the original style, ensuring historical accuracy, and adhering to the "
#             "linguistic conventions of the 18th century.[/INST]"
#             "\n\n## Guidelines:"
#             "\n- Address only OCR errors; please do not add more content."
#             "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
#             "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
#             "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
#             "\n\n## Examples of Corrections:"
#             "\n1. OCR Error from the User: '30th. Letters Orders and Instructions December 1755.' "
#             "Correction from the assistant: '308th Letters, Orders, and Instructions, December 1755.'"
#             "\n2. OCR Error from the User: 'remain here until the arrival of the usual with' "
#             "Correction from the assistant: 'remain here until the arrival of the vessel with'"
#             "\n3. OCR Error from the User: 'thes, Vc. and to me directions' "
#             "Correction from the assistant: 'the Stores, &c., and to be under the same directions'"
#             "\n4. OCR Error from the User: 'of Clothes; Shoes, Stockings, Shirts, Vc.: propatioma-' "
#             "Correction from the assistant: 'of Clothes; Shoes, Stockings, Shirts, &c.: proportionate-'"
#             "\n5. OCR Error from the User: 'things delivered into into the Nuggets, and are that' "
#             "Correction from the assistant: 'Things delivered into the wagons, and see that'"
#             "\n6. OCR Error from the User: 'No. To Doctor James Frank of the Virginia' "
#             "Correction from the assistant: 'To Doctor James Craik, of the Virginia'"
#             "\n7. OCR Error from the User: 'as before ordered. To soon as the Stores arrive, you' "
#             "Correction from the assistant: 'as before ordered. As soon as the Stores arrive, you'"
#             "\n8. OCR Error from the User: 'You are are immediately, upon receipt' "
#             "Correction from the assistant: 'You are immediately, upon receipt'"
#             "\n9. OCR Error from the User: 'hereof, to repair to Winchester, where you will never' "
#             "Correction from the assistant: 'hereof, to repair to Winchester, where you will meet'"
#             "\n10. OCR ERROR from the User: 'you that that, and processing, yourself to Winches-'"
#             "Correction from the assistant: 'your Chest, and proceeding, yourself, to Winches-'"
#             "</s>"
#         )
#     else:
#         nummer_length = (len(sentence) * 2) + 2100
#         system_prompt = (
#             "<s>"
#             "[INST] Task: Correct OCR errors found in a dataset of 18th-century documents. These errors range from simple "
#             "misspellings to complex issues like incorrect abbreviations and term misinterpretations. Your objective is "
#             "to correct these errors with precision, ensuring the integrity of the original manuscripts is maintained. "
#             "Avoid introducing new information or omitting essential details Focus on retaining the original style, "
#             "accuracy, and 18th-century linguistic conventions.[/INST]"
#             "\n\n## Guidelines:"
#             "\n- Address only OCR errors; please do not add more content."
#             "\n- Ensure corrections accurately reflect the 18th-century language, style, and conventions."
#             "\n- If you find cut-out words at the end of the OCR sentence don't complete them."
#             "\n- If the corrected sentence is twice the length of the OCR sentence, return the OCR sentence. "
#             "\n\n## Examples of corrected sentences modelling cases:"
#             "\n1. OCR Error from the User: '30th. Letters Orders and Instructions December 1755.' "
#             "Correction from the assistant: '308th Letters, Orders, and Instructions, December 1755.'"
#             "\n2. OCR Error from the User: 'remain here until the arrival of the usual with' "
#             "Correction from the assistant: 'remain here until the arrival of the vessel with'"
#             "\n3. OCR Error from the User: 'thes, Vc. and to me directions' "
#             "Correction from the assistant: 'the Stores, &c., and to be under the same directions'"
#             "\n4. OCR Error from the User: 'of Clothes; Shoes, Stockings, Shirts, Vc.: propatioma-' "
#             "Correction from the assistant: 'of Clothes; Shoes, Stockings, Shirts, &c.: proportionate-'"
#             "\n5. OCR Error from the User: 'things delivered into into the Nuggets, and are that' "
#             "Correction from the assistant: 'Things delivered into the wagons, and see that'"
#             "\n6. OCR Error from the User: 'No. To Doctor James Frank of the Virginia' "
#             "Correction from the assistant: 'To Doctor James Craik, of the Virginia'"
#             "\n7. OCR Error from the User: 'as before ordered. To soon as the Stores arrive, you' "
#             "Correction from the assistant: 'as before ordered. As soon as the Stores arrive, you'"
#             "\n8. OCR Error from the User: 'You are are immediately, upon receipt' "
#             "Correction from the assistant: 'You are immediately, upon receipt'"
#             "\n9. OCR Error from the User: 'hereof, to repair to Winchester, where you will never' "
#             "Correction from the assistant: 'hereof, to repair to Winchester, where you will meet'"
#             "\n10. OCR ERROR from the User: 'you that that, and processing, yourself to Winches-' "
#             "Correction from the assistant: 'your Chest, and proceeding, yourself, to Winches-'"
#             "\n11. OCR from the User: 'percent number of Waggons to carry them to Win-' "
#             "Correction from the assistant: 'ficient number of waggons to carry them to Win-'"
#             "\n12. OCR from the User: 'an Shirt Guard is kept our them, that no undes-' "
#             "Correction from the assistant: 'a strict Guard is kept over them, that no embez-'"
#             "\n13. OCR from the User: 'concerning Limited, Limited on their march upheros, there' "
#             "Correction from the assistant: 'conveniently Quartered on their march up; there'"
#             "\n14. OCR from the User: 'must be made of. If any of the Soldiers' "
#             "Correction from the assistant: 'must be made use of. - If any of the Soldiers'"
#             "\n15. OCR from the User: 'should direct front from this place, if would be proper' "
#             "Correction from the assistant: 'should desert from this place, - it would be proper'"
#             "\n16. PCR from the User: 'eds until for Duty, or Duty, on Review, are ordered to be' "
#             "Correction from the assistant: 'ed unfit for Duty, on Review; are ordered to be'"
#             "\n17. OCR from the User: 'up the Cobbs by the first pregnant' "
#             "Correction from the assistant: 'up the Cask of Tools by the first waggon that'"
#             "\n18. OCR from the User: 'the aid de camp. We Mr. Mr. We knowingly is to send' "
#             "Correction from the assistant: 'the aid de camp. The Commissary is to send'"
#             "\n19. OCR from the User: 'Colonel, Stephen will give you you an ac ac-' "
#             "Correction from the assistant: 'Colonel Stephen will give you an ac-'"
#             "\n20. OCR from the User: 'commissioned officered Officer or Solution should happen to' "
#             "Correction from the assistant: 'commissioned Officer or Soldier should happen to'"
#             "\n21. OCR from the User: 'You are to be particularly ex-' "
#             "Correction from the assistant: 'You are to be particularly ex-'"
#             "\n22. OCR from the User: 'that meeting with Letters at Fredericks-'"
#             "Correction from the assistant: 'that meeting with Letters at Fredericks-'"
#             "\n23. OCR from the User: 'proper Dispositions, and seeing that all our neces-' "
#             "Correction from the assistant: 'proper Dispositions, and seeing that all our neces-'"
#             "\n24 OCR from the User: 'have a very good effect. The Commonalty in gene-' "
#             "Correction from the assistant: 'have a very good effect. The Commonalty in gene-'"
#             "\n25. OCR from the User: 'money are greatest. When I left Willi-' "
#             "Correction from the assistant: 'money are greatest. When I left Willi-'"
#             "\n26. OCR from the User: 'with sundry arms, Vc. for the Troops. This Ves-' "
#             "Correction from the assistant: 'with sundry arms, Vc. for the Troops. This Ves-'"
#             "\n27. OCR from the User: 'me much concern; fearing that that this de-' "
#             "Correction from the assistant: 'me much concern; fearing that this de-'"
#             "\n28. OCR from the User: 'the Spring. If your Honor think proper to or-' "
#             "Correction from the assistant: 'the Spring. If your Honor think proper to or-'"
#             "\n29. OCR from the User: 'would fix it: so is Captain Stewarts: If Captain Stew-' "
#             "Correction from the assistant: 'would fix it: so is Captain Stewarts: If Captain Stew-'"
#             "\n30. OCR from the User: 'proper Dispositions, and seeing that all our neces-' "
#             "Correction from the assistant: 'proper Dispositions, and seeing that all our neces-'"
#             "\n31. OCR from the User: 'Letters Orders Orders and Instructions. Decembers. December 1755.' "
#             "Correction from the assistant: 'Letters Orders and Instructions December. 1755.'"
#             "\n32. OCR from the User: 'There importantly expecte' "
#             "Correction from the assistant: 'I have impatiently expected'"
#             "\n33. OCR from the User: 'the Ist. of December, to Rendezvous at Alexa-' "
#             "Correction from the assistant: 'the 1st. of December, to Rendezvous at Alex-'"
#             "</s>"
#         )
#
#     adaptation_request = (
#         f"<s>"
#         f"[INST] Based on the guidelines and illustrated examples, accurately correct the OCR errors in the following "
#         f"sentence without adding extraneous information: '{sentence}'. If you don't know how to correct return the "
#         f"original sentence as is: '{sentence}'.[/INST]"
#         f"</s>"
#     )
#
#     prompt = f"{system_prompt}\n{adaptation_request}\nThe corrected sentence from the assistant:"
#     # nummer_length = (len(sentence) * 2) + 2100
#
#     try:
#         corrected_text = calculate_pipe(pipe, prompt, nummer_length, 1)
#         response = corrected_text[0]['generated_text'].split('The corrected sentence from the assistant:')[-1].strip()
#         # Post-processing to remove any additional unwanted text
#         response = response[1:-1]
#         response = response.replace("'", "")
#         response = response.split('\n')[0].strip()
#
#         if (len(response)) > (len(sentence)*2):
#             print(f"The response from MISTRAL is very long: {response}")
#             response = sentence
#
#     except Exception as e:
#         print(f"Error in processing sentence '{sentence}': {e}")
#         response = sentence
#     return response


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

