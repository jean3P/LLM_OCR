import os
import sys

abs_path = sys.path[0]
TOKEN = "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"
base_name = os.path.dirname(abs_path)
resources_path = os.path.join('./../resources')
washington_path = os.path.join(resources_path, 'washingtondb-v1.0')
outputs_path = os.path.join(resources_path, 'outputs')
iam_outputs_path = os.path.join(outputs_path, 'IAM_dataset')
iam_path = os.path.join(resources_path, 'IAM-V1')
transcription_washington_path = os.path.join(washington_path, 'ground_truth', 'transcription.txt')
transcription_iam_path = os.path.join(iam_path, 'ground_truth', 'lines.txt')
results_test_trocr = os.path.join(outputs_path, 'results_test')
results_LLM_mistral = os.path.join(outputs_path, 'results_LLM')
results_LLM_mistral_1 = os.path.join(outputs_path, 'results_LLM_Mistral7B_1')
results_LLM_mistral_2 = os.path.join(outputs_path, 'results_LLM_Mistral7B_1_v2')
results_LLM_mistral_3 = os.path.join(outputs_path, 'results_LLM_Mistral7B_1_v3')
automated_resuts = os.path.join(outputs_path, 'automated_results')
automated_resuts_experiments = os.path.join(outputs_path, 'experiments', 'automated_results')
pipeline_v1_path = os.path.join(outputs_path, 'pipeline_mistral_v1')
pipeline_v1_mistral_path = os.path.join(pipeline_v1_path, 'mistral')
pipeline_v1_mistral_ocr = os.path.join(pipeline_v1_path, 'ocr')

outputs_path_test = os.path.join(outputs_path, 'test')
#

results_mixed_LLM_MISTRAL = os.path.join(outputs_path, 'results_mixed_OCR_MISTRAL')

# Specify the directory to save the model
model_save_path = os.path.join(outputs_path, 'model', 'trained_trocr_model')
processor_save_path = os.path.join(outputs_path, 'model', 'trocr_processor')

model_save_path_seq = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq')
model_save_path_seq_v2 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2')
processor_save_path_seq = os.path.join(outputs_path, 'model', 'trocr_processor_seq')
processor_save_path_seq_v2 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2')

model_save_path_seq_mixed_20_80 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_mixed_20_80')
model_save_path_seq_v2_mixed_20_80 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_mixed_20_80')
processor_save_path_seq_mixed_20_80 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_mixed_20_80')
processor_save_path_seq_v2_mixed_20_80 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_mixed_20_80')

# ======
model_save_path_seq_25 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_25')
model_save_path_seq_v2_25 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_25')
processor_save_path_seq_25 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_25')
processor_save_path_seq_v2_25 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_25')

model_save_path_seq_25_75 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_25_75')
model_save_path_seq_v2_25_75 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_25_75')
processor_save_path_seq_25_75 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_25_75')
processor_save_path_seq_v2_25_75 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_25_75')

model_save_path_seq_50 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_50')
model_save_path_seq_v2_50 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_50')
processor_save_path_seq_50 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_50')
processor_save_path_seq_v2_50 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_50')

model_save_path_seq_50_50 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_50_50')
model_save_path_seq_v2_50_50 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_50_50')
processor_save_path_seq_50_50 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_50_50')
processor_save_path_seq_v2_50_50 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_50_50')

model_save_path_seq_75 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_75')
model_save_path_seq_v2_75 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_75')
processor_save_path_seq_75 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_75')
processor_save_path_seq_v2_75 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_75')

model_save_path_seq_75_25 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_75_25')
model_save_path_seq_v2_75_25 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_75_25')
processor_save_path_seq_75_25 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_75_25')
processor_save_path_seq_v2_75_25 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_75_25')

model_save_path_seq_100 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_100')
model_save_path_seq_v2_100 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2_100')
processor_save_path_seq_100 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_100')
processor_save_path_seq_v2_100 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2_100')

# =======
REPLACEMENTS_WASHINGTON = {
    's_pt': '.', 's_cm': ',', 's_mi': '-', 's_sq': ";", 's_dash': '-',
    's_sl': '/', 's_bsl': '\\', 's_qm': '?', 's_exc': '!', 's_col': ':',
    's_sc': ';', 's_lp': '(', 's_rp': ')', 's_lb': '[', 's_rb': ']',
    's_lc': '{', 's_rc': '}', 's_dq': '"', 's_ap': '@', 's_hs': '#',
    's_dl': '$', 's_pc': '%', 's_am': '&', 's_ast': '*', 's_pl': '+',
    's_eq': '=', 's_lt': '<', 's_gt': '>', 's_us': '_', 's_crt': '^',
    's_tld': '~', 's_vbar': '|', 's_sp': ' ', 's_s': 's', 's_qt': "'",
    's_GW': 'G.W.', 's_qo': ':', 's_et': 'V', 's_br': ')', 's_bl': '(',
    '|': " ", '-': '', 's_': ''
}

WASHINGTON_SAMPLE_TEXT = ("270. Letters, Orders and Instructions. October 1755.\n"
                          "only for the publick use, unless by particu-\n"
                          "lar Orders from me. You are to send\n"
                          "down a Barrel of Flints with the Arms, to\n"
                          "Winchester, and about two thousand weight\n"
                          "of Flour, for the two Companies of Rangers;\n"
                          "twelve hundred of which to be delivered\n"
                          "Captain Ashby and Company, at the\n"
                          "Plantation of Charles Sellars - the rest to Captain\n"
                          "Cockes' Company, at Nicholas Reasmers.\n"
                          "October 26th. G.W.\n"
                          "28th Winchester: October 28th, 1755.\n"
                          "Parole Hampton.\n"
                          "The Officers who came down\n"
                          "from Fort Cumberland with Colonel\n"
                          "Washington, are immediately to go Recrui-\n"
                          "ting; and they are allowed until the 1st. of De-\n"
                          "cember; at which time if they do not\n"
                          "punctually appear at the place of Rendez-\n"
                          "vous assigned them, they will be tried by a\n"
                          "Court Martial, for disobedience of Orders.\n"
                          "They are to wait upon the Aid de camp\n"
                          "at one of the Clock, to receive their Recrui-\n"
                          "ting Instructions. Each Officer present, to give\n"
                          "in a Return immediately of the number\n"
                          "of men he has enlisted. - One Subaltern,\n"
                          "one Sergeant, one Corporal, one Drummer,\n"
                          "and twenty five private men, are to mount\n"
                          "Guard to - day, and to be relieved to - morrow\n"
                          "at ten o'clock. -. All Reports and Returns\n"
                          "are to be made to the Aid de Camp.")

EXAMPLES = (
    '\n1. OCR Error from the User: 20%. Letterss orders and Instructions and Instructions Decembers December 1755.'
    '   The JSON output is: '
    '     Corrected sentence: 308. Letters Orders and Instructions December 1755. '
    '     Percentage of confidence: 61% '
    '     Justification: Corrected the misrecognition 20% to 308 and removed repeated phrases. '
    'The relatively lower confidence indicates significant recognition errors.'
    '\n2. OCR Error from the User: remain here until the arrival of the visit with '
    '   The JSON output is: '
    '     Corrected sentence: remain here until the arrival of the vessel with '
    '     Percentage of confidence: 92% '
    '     Justification: Corrected the misrecognition visit to vessel. The high confidence '
    'indicates most of the text was correctly recognized.'
    '\n3. OCR Error from the User: the Flours, Vc. and to be under the same directions '
    '   The JSON output is: '
    '     Corrected sentence: the Stores, Vc. and to be under the same directions '
    '     Percentage of confidence: 92% '
    '     Justification: Corrected the misrecognition Flours to Stores. The high confidence suggests '
    'accurate recognition for most parts.'
    '\n4. OCR Error from the User: as to for ordered. To soon as as the Stores arrive, you '
    '   The JSON output is: '
    '     Corrected sentence: as before ordered. So soon as the Stores arrive, you '
    '     Percentage of confidence: 85% '
    '     Justification: Corrected the misrecognition to for to before and removed repeated phrases. '
    'The high confidence indicates good recognition accuracy.'
    '\n5. OCR Error from the User: failed number of suggestions to carry them to Your- '
    '   The JSON output is: '
    '     Corrected sentence: ficient number of waggons to carry them to Win- '
    '     Percentage of confidence: 71% '
    '     Justification: Corrected failed number of suggestions to ficient number of waggons and adjusted '
    'the phrase. The confidence level indicates some errors.'
    '\n6. OCR Error from the User: whatever; whether they they are to be best, under the '
    '   The JSON output is: '
    '     Corrected sentence: chester; whither they are to be sent, under the '
    '     Percentage of confidence: 75% '
    '     Justification: Corrected whatever; whether to chester; whither and removed repeated words. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n7. OCR Error from the User: donut of the Soldiers now now here: except the Suits but '
    '   The JSON output is: '
    '     Corrected sentence: escort of the Soldiers now here: except the Suits '
    '     Percentage of confidence: 77% '
    '     Justification: Corrected donut to escort and removed repeated phrases. The confidence level indicates '
    'some parts were correctly identified.'
    '\n8. OCR Error from the User: of clothes, Shoes, Stains, Shirts, buts, promotional- '
    '   The JSON output is: '
    '     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona- '
    '     Percentage of confidence: 74% '
    '     Justification: Corrected Stains to Stocking and adjusted the phrase. The confidence level '
    'indicates some recognition errors.'
    '\n9. OCR Error from the User: ity, which we are to be to be left with both lobbied Carlylely. '
    '   The JSON output is: '
    '     Corrected sentence: bly which are to be left with Colonel Carlyle. '
    '     Percentage of confidence: 65% '
    '     Justification: Corrected ity to bly and removed repeated phrases. The lower confidence '
    'indicates significant recognition errors.'
    '\n10. OCR Error from the User: somes and medicines arrive, you are are to smokes. '
    '   The JSON output is: '
    '     Corrected sentence: Stores and medicines arrive, you are to embrace '
    '     Percentage of confidence: 72% '
    '     Justification: Corrected somes to Stores and removed repeated words. The confidence level '
    'indicates moderate recognition accuracy.'
    '\n11. OCR Error from the User: 30th. Letters Orders and Instructions December 1755. '
    '   The JSON output is: '
    '     Corrected sentence: 308. Letters Orders and Instructions December 1755. '
    '     Percentage of confidence: 96% '
    '     Justification: Corrected the misrecognition 30th to 308. The high confidence indicates the '
    'rest of the text was accurately recognized.'
    '\n12. OCR Error from the User: remain her animal of the usual with '
    '   The JSON output is: '
    '     Corrected sentence: remain here until the arrival of the vessel with '
    '     Percentage of confidence: 62% '
    '     Justification: Corrected her animal to here until the arrival and usual to vessel. The lower confidence '
    'indicates significant recognition errors.'
    '\n13. OCR Error from the User: the Box, V. and the same directions '
    '   The JSON output is: '
    '     Corrected sentence: the Stores, Vc. and to be under the same directions '
    '     Percentage of confidence: 65% '
    '     Justification: Corrected Box, V. to Stores, Vc. and adjusted the phrase. The confidence level '
    'indicates some recognition errors.'
    '\n14. OCR Error from the User: are, with all possible dispatch, to encourage a suf- '
    '   The JSON output is: '
    '     Corrected sentence: are, with all possible dispatch, to procure a suf- '
    '     Percentage of confidence: 87% '
    '     Justification: Corrected encourage to procure. The high confidence suggests most of the text was '
    'correctly recognized.'
    '\n15. OCR Error from the User: fact number of Waggons to carry them to Him- '
    '   The JSON output is: '
    '     Corrected sentence: ficient number of waggons to carry them to Win- '
    '     Percentage of confidence: 85% '
    '     Justification: Corrected fact to ficient and Him to Win. The high confidence indicates good '
    'recognition accuracy.'
    '\n16. OCR Error from the User: of Octobers, Ibids, Vc., proportion- '
    '   The JSON output is: '
    '     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona- '
    '     Percentage of confidence: 47% '
    '     Justification: Corrected Octobers, Ibids to Clothes; Shoes, Stocking, Shirts. The lower confidence '
    'indicates significant recognition errors.'
    '\n17. OCR Error from the User: they, which are to be to be left with with Colonial Carlyle. '
    '   The JSON output is: '
    '     Corrected sentence: bly which are to be left with Colonel Carlyle. '
    '     Percentage of confidence: 72% '
    '     Justification: Removed repeated phrases and corrected Colonial to Colonel. The confidence level '
    'indicates some parts were correctly identified.'
    '\n18. OCR Error from the User: a joint Guard is kept over them, that us unless '
    '   The JSON output is: '
    '     Corrected sentence: a strict Guard is kept over them, that no embez- '
    '     Percentage of confidence: 77% '
    '     Justification: Corrected joint to strict and us unless to no embez-. The confidence level '
    'indicates moderate recognition accuracy.'
    '\n19. OCR Error from the User: conveniently Burntered on their march up; thee '
    '   The JSON output is: '
    '     Corrected sentence: conveniently Quartered on their march up; there '
    '     Percentage of confidence: 91% '
    '     Justification: Corrected Burntered to Quartered and thee to there. The high confidence suggests most '
    'of the text was correctly recognized.'
    '\n20. OCR Error from the User: To. To Captain John John John Mercer. of the '
    '   The JSON output is: '
    '     Corrected sentence: 16th. To Captain John Mercer. of the '
    '     Percentage of confidence: 68% '
    '     Justification: Corrected repeated words To Captain John John John to 16th. To Captain John Mercer. '
    'The confidence level indicates some recognition errors.'
    '\n21. OCR Error from the User: remain until the issue with '
    '   The JSON output is: '
    '     Corrected sentence: remain here until the arrival of the vessel with '
    '     Percentage of confidence: 52% '
    '     Justification: Corrected remain until the issue to remain here until the arrival and issue to vessel. '
    'The low confidence indicates significant recognition errors.'
    '\n22. OCR Error from the User: the House, Vc be under some directions '
    '   The JSON output is: '
    '     Corrected sentence: the Stores, Vc. and to be under the same directions '
    '     Percentage of confidence: 65% '
    '     Justification: Corrected House to Stores, added missing and, and corrected some to same. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n23. OCR Error from the User: ficial number number of Waggons to carry them to Win- '
    '   The JSON output is: '
    '     Corrected sentence: ficient number of waggons to carry them to Win- '
    '     Percentage of confidence: 83% '
    '     Justification: Corrected ficial to ficient and removed repeated number. '
    'The high confidence indicates good recognition accuracy.'
    '\n24. OCR Error from the User: escorit of the Soldiers now here: except the Suits '
    '   The JSON output is: '
    '     Corrected sentence: escort of the Soldiers now here: except the Suits '
    '     Percentage of confidence: 98% '
    '     Justification: Corrected escorit to escort. The high confidence indicates most of '
    'the text was correctly recognized.'
    '\n25. OCR Error from the User: bly, which are to be to be left with Sociali. '
    '   The JSON output is: '
    '     Corrected sentence: bly which are to be left with Colonel Carlyle. '
    '     Percentage of confidence: 59% '
    '     Justification: Corrected Sociali to Colonel Carlyle and removed repeated to be. '
    'The low confidence indicates significant recognition errors.'
    '\n26. OCR Error from the User: a visit is kept them, that no - - unless- '
    '   The JSON output is: '
    '     Corrected sentence: a strict Guard is kept over them, that no embez- '
    '     Percentage of confidence: 50% '
    '     Justification: Corrected a visit is kept them to a strict Guard is kept over them and corrected '
    'unless- to embez-. The low confidence indicates significant recognition errors.'
    '\n27. OCR Error from the User: fement is made. If your men men can not be '
    '   The JSON output is: '
    '     Corrected sentence: zlement is made. If your men can not be '
    '     Percentage of confidence: 86% '
    '     Justification: Corrected fement to zlement and removed repeated men. '
    'The high confidence indicates most of the text was correctly recognized.'
    '\n28. OCR Error from the User: community Quarterly Quartered on their march up; there '
    '   The JSON output is: '
    '     Corrected sentence: conveniently Quartered on their march up; there '
    '     Percentage of confidence: 78% '
    '     Justification: Corrected community Quarterly to conveniently and removed redundant Quarterly. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n29. OCR Error from the User: Given a technical Vc. '
    '   The JSON output is: '
    '     Corrected sentence: Given at Alexandria Vc. '
    '     Percentage of confidence: 61% '
    '     Justification: Corrected a technical to at Alexandria. The lower confidence '
    'indicates significant recognition errors.'
    '\n30. OCR Error from the User: to. To Doctor James I amsts Crash of the Virginia '
    '   The JSON output is: '
    '     Corrected sentence: 16th. To Doctor James Craik, of the Virginia '
    '     Percentage of confidence: 71% '
    '     Justification: Corrected to. To to 16th. To and I amsts Crash to Doctor James Craik. '
    'The moderate confidence indicates recognition errors.'
    "</s>")

EXAMPLES_20 = (
    '\n1. OCR Error from the User: 20%. Letterss orders and Instructions and Instructions Decembers December 1755.'
    '   The JSON output is: '
    '     Corrected sentence: 308. Letters Orders and Instructions December 1755. '
    '     Percentage of confidence: 61% '
    '     Justification: Corrected the misrecognition 20% to 308 and removed repeated phrases. '
    'The relatively lower confidence indicates significant recognition errors.'
    '\n2. OCR Error from the User: remain here until the arrival of the visit with '
    '   The JSON output is: '
    '     Corrected sentence: remain here until the arrival of the vessel with '
    '     Percentage of confidence: 92% '
    '     Justification: Corrected the misrecognition visit to vessel. The high confidence '
    'indicates most of the text was correctly recognized.'
    '\n3. OCR Error from the User: the Flours, Vc. and to be under the same directions '
    '   The JSON output is: '
    '     Corrected sentence: the Stores, Vc. and to be under the same directions '
    '     Percentage of confidence: 92% '
    '     Justification: Corrected the misrecognition Flours to Stores. The high confidence suggests '
    'accurate recognition for most parts.'
    '\n4. OCR Error from the User: as to for ordered. To soon as as the Stores arrive, you '
    '   The JSON output is: '
    '     Corrected sentence: as before ordered. So soon as the Stores arrive, you '
    '     Percentage of confidence: 85% '
    '     Justification: Corrected the misrecognition to for to before and removed repeated phrases. '
    'The high confidence indicates good recognition accuracy.'
    '\n5. OCR Error from the User: failed number of suggestions to carry them to Your- '
    '   The JSON output is: '
    '     Corrected sentence: ficient number of waggons to carry them to Win- '
    '     Percentage of confidence: 71% '
    '     Justification: Corrected failed number of suggestions to ficient number of waggons and adjusted '
    'the phrase. The confidence level indicates some errors.'
    '\n6. OCR Error from the User: whatever; whether they they are to be best, under the '
    '   The JSON output is: '
    '     Corrected sentence: chester; whither they are to be sent, under the '
    '     Percentage of confidence: 75% '
    '     Justification: Corrected whatever; whether to chester; whither and removed repeated words. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n7. OCR Error from the User: donut of the Soldiers now now here: except the Suits but '
    '   The JSON output is: '
    '     Corrected sentence: escort of the Soldiers now here: except the Suits '
    '     Percentage of confidence: 77% '
    '     Justification: Corrected donut to escort and removed repeated phrases. The confidence level indicates '
    'some parts were correctly identified.'
    '\n8. OCR Error from the User: of clothes, Shoes, Stains, Shirts, buts, promotional- '
    '   The JSON output is: '
    '     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona- '
    '     Percentage of confidence: 74% '
    '     Justification: Corrected Stains to Stocking and adjusted the phrase. The confidence level '
    'indicates some recognition errors.'
    '\n9. OCR Error from the User: ity, which we are to be to be left with both lobbied Carlylely. '
    '   The JSON output is: '
    '     Corrected sentence: bly which are to be left with Colonel Carlyle. '
    '     Percentage of confidence: 65% '
    '     Justification: Corrected ity to bly and removed repeated phrases. The lower confidence '
    'indicates significant recognition errors.'
    '\n10. OCR Error from the User: somes and medicines arrive, you are are to smokes. '
    '   The JSON output is: '
    '     Corrected sentence: Stores and medicines arrive, you are to embrace '
    '     Percentage of confidence: 72% '
    '     Justification: Corrected somes to Stores and removed repeated words. The confidence level '
    'indicates moderate recognition accuracy.'
    '\n11. OCR Error from the User: 30th. Letters Orders and Instructions December 1755. '
    '   The JSON output is: '
    '     Corrected sentence: 308. Letters Orders and Instructions December 1755. '
    '     Percentage of confidence: 96% '
    '     Justification: Corrected the misrecognition 30th to 308. The high confidence indicates the '
    'rest of the text was accurately recognized.'
    '\n12. OCR Error from the User: remain her animal of the usual with '
    '   The JSON output is: '
    '     Corrected sentence: remain here until the arrival of the vessel with '
    '     Percentage of confidence: 62% '
    '     Justification: Corrected her animal to here until the arrival and usual to vessel. The lower confidence '
    'indicates significant recognition errors.'
    '\n13. OCR Error from the User: the Box, V. and the same directions '
    '   The JSON output is: '
    '     Corrected sentence: the Stores, Vc. and to be under the same directions '
    '     Percentage of confidence: 65% '
    '     Justification: Corrected Box, V. to Stores, Vc. and adjusted the phrase. The confidence level '
    'indicates some recognition errors.'
    '\n14. OCR Error from the User: are, with all possible dispatch, to encourage a suf- '
    '   The JSON output is: '
    '     Corrected sentence: are, with all possible dispatch, to procure a suf- '
    '     Percentage of confidence: 87% '
    '     Justification: Corrected encourage to procure. The high confidence suggests most of the text was '
    'correctly recognized.'
    '\n15. OCR Error from the User: fact number of Waggons to carry them to Him- '
    '   The JSON output is: '
    '     Corrected sentence: ficient number of waggons to carry them to Win- '
    '     Percentage of confidence: 85% '
    '     Justification: Corrected fact to ficient and Him to Win. The high confidence indicates good '
    'recognition accuracy.'
    '\n16. OCR Error from the User: of Octobers, Ibids, Vc., proportion- '
    '   The JSON output is: '
    '     Corrected sentence: of Clothes; Shoes, Stocking, Shirts, Vc. proportiona- '
    '     Percentage of confidence: 47% '
    '     Justification: Corrected Octobers, Ibids to Clothes; Shoes, Stocking, Shirts. The lower confidence '
    'indicates significant recognition errors.'
    '\n17. OCR Error from the User: they, which are to be to be left with with Colonial Carlyle. '
    '   The JSON output is: '
    '     Corrected sentence: bly which are to be left with Colonel Carlyle. '
    '     Percentage of confidence: 72% '
    '     Justification: Removed repeated phrases and corrected Colonial to Colonel. The confidence level '
    'indicates some parts were correctly identified.'
    '\n18. OCR Error from the User: a joint Guard is kept over them, that us unless '
    '   The JSON output is: '
    '     Corrected sentence: a strict Guard is kept over them, that no embez- '
    '     Percentage of confidence: 77% '
    '     Justification: Corrected joint to strict and us unless to no embez-. The confidence level '
    'indicates moderate recognition accuracy.'
    '\n19. OCR Error from the User: conveniently Burntered on their march up; thee '
    '   The JSON output is: '
    '     Corrected sentence: conveniently Quartered on their march up; there '
    '     Percentage of confidence: 91% '
    '     Justification: Corrected Burntered to Quartered and thee to there. The high confidence suggests most '
    'of the text was correctly recognized.'
    '\n20. OCR Error from the User: To. To Captain John John John Mercer. of the '
    '   The JSON output is: '
    '     Corrected sentence: 16th. To Captain John Mercer. of the '
    '     Percentage of confidence: 68% '
    '     Justification: Corrected repeated words To Captain John John John to 16th. To Captain John Mercer. '
    'The confidence level indicates some recognition errors.'
)

EXAMPLES_F = (
    '\n1. OCR Error from the User: to the Return made at - Food- Coun-'
    '   The JSON output is: '
    '     Corrected sentence: to the Return made me at Fort Cum- '
    '     Percentage of confidence: 74% '
    '     Justification: Corrected the misrecognition Food- Coun- to Fort Cum-. '
    'The confidence level indicates some recognition errors.'
    '\n2. OCR Error from the User: Ireland, October, - I have sent more'
    '   The JSON output is: '
    '     Corrected sentence: berland, October 26th. - I have sent more '
    '     Percentage of confidence: 78% '
    '     Justification: Corrected the misrecognition Ireland, October to October 26th. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n3. OCR Error from the User: Suits than are necessary, in case of options'
    '   The JSON output is: '
    '     Corrected sentence: Suits than are necessary, in case of getting '
    '     Percentage of confidence: 89% '
    '     Justification: Corrected the misrecognition options to getting. '
    'The high confidence suggests accurate recognition for most parts.'
    '\n4. OCR Error from the User: a Recruit to. You must do the best you'
    '   The JSON output is: '
    '     Corrected sentence: a Recruit Vc. You must do the best you '
    '     Percentage of confidence: 95% '
    '     Justification: Corrected the misrecognition to to Vc. '
    'The high confidence indicates good recognition accuracy.'
    '\n5. OCR Error from the User: can indoing the men, as the the Merchants.'
    '   The JSON output is: '
    '     Corrected sentence: can in Lodging the men, as the Barracks '
    '     Percentage of confidence: 64% '
    '     Justification: Corrected indoing the men to in Lodging the men and removed repeated words. '
    'The confidence level indicates moderate recognition errors.'
    '\n6. OCR Error from the User: in the Hot are full. It\'s Sergeant Will -'
    '   The JSON output is: '
    '     Corrected sentence: in the Fort are full. As Sergeant Wil- '
    '     Percentage of confidence: 83% '
    '     Justification: Corrected Hot to Fort and It\'s to As. '
    'The confidence level indicates good recognition accuracy.'
    '\n7. OCR Error from the User: ger is waiting the Return of the the Keegan\'s'
    '   The JSON output is: '
    '     Corrected sentence: per is waiting the return of the waggons '
    '     Percentage of confidence: 76% '
    '     Justification: Corrected ger to per and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n8. OCR Error from the User: from the Fort for fornecaries for Captain Captain'
    '   The JSON output is: '
    '     Corrected sentence: from the Fort, for necessaries for Captain '
    '     Percentage of confidence: 69% '
    '     Justification: Corrected fornecaries to necessaries and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n9. OCR Error from the User: Rogues Company, so soon as as they arrive,'
    '   The JSON output is: '
    '     Corrected sentence: Hoggs Company; so soon as they arrive, '
    '     Percentage of confidence: 83% '
    '     Justification: Corrected Rogues to Hoggs and removed repeated words. '
    'The confidence level indicates good recognition accuracy.'
    '\n10. OCR Error from the User: you are to see to see that he receives such'
    '   The JSON output is: '
    '     Corrected sentence: you are to see that he receives such '
    '     Percentage of confidence: 84% '
    '     Justification: Removed repeated words. '
    'The confidence level indicates good recognition accuracy.'
    '\n11. OCR Error from the User: things as he has been for, andathah'
    '   The JSON output is: '
    '     Corrected sentence: things as he has orders for, and dispatch '
    '     Percentage of confidence: 71% '
    '     Justification: Corrected been for to orders for and corrected andathah to dispatch. '
    'The confidence level indicates some recognition errors.'
    '\n12. OCR Error from the User: him immediately. If no no other Morses, debug.'
    '   The JSON output is: '
    '     Corrected sentence: him immediately. If no other Horses, belong- '
    '     Percentage of confidence: 80% '
    '     Justification: Corrected no no to no and Morses, debug to Horses, belong-. '
    'The confidence level indicates some recognition errors.'
    '\n13. OCR Error from the User: engagee the her heremen to remain in remain with'
    '   The JSON output is: '
    '     Corrected sentence: engage the herdsmen to remain with '
    '     Percentage of confidence: 67% '
    '     Justification: Corrected engagee the her heremen to engage the herdsmen and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n14. OCR Error from the User: injury, or from me me. Now must engage'
    '   The JSON output is: '
    '     Corrected sentence: missary or from me. You must engage '
    '     Percentage of confidence: 74% '
    '     Justification: Corrected injury, or from me me to missary or from me. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n15. OCR Error from the User: all the Coopers you can can to make 1sts.'
    '   The JSON output is: '
    '     Corrected sentence: all the Coopers you can to make Barrels '
    '     Percentage of confidence: 73% '
    '     Justification: Removed repeated words and corrected 1sts to Barrels. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n16. OCR Error from the User: for packing the Beef, Beef, and if any of the'
    '   The JSON output is: '
    '     Corrected sentence: for packing the Beef; and if any of the '
    '     Percentage of confidence: 84% '
    '     Justification: Removed repeated words. '
    'The confidence level indicates good recognition accuracy.'
    '\n17. OCR Error from the User: enchiladaly to work. You are to receive'
    '   The JSON output is: '
    '     Corrected sentence: mediately to work. You are to receive '
    '     Percentage of confidence: 82% '
    '     Justification: Corrected enchiladaly to immediately. '
    'The confidence level indicates some recognition errors.'
    '\n18. OCR Error from the User: Roasts, short, which you are will deliver to'
    '   The JSON output is: '
    '     Corrected sentence: Goose - which you will deliver to '
    '     Percentage of confidence: 66% '
    '     Justification: Corrected Roasts, short, to Goose -. '
    'The confidence level indicates some recognition errors.'
    '\n19. OCR Error from the User: Captain thinking as you pege'
    '   The JSON output is: '
    '     Corrected sentence: Captain Ashby\'s Company as you pass '
    '     Percentage of confidence: 54% '
    '     Justification: Corrected thinking as you pege to Ashby\'s Company as you pass. '
    'The confidence level indicates significant recognition errors.'
    '\n20. OCR Error from the User: directing the Netherlands.'
    '   The JSON output is: '
    '     Corrected sentence: by, directing him to be particularly care- '
    '     Percentage of confidence: 48% '
    '     Justification: Corrected directing the Netherlands to directing him to be particularly care-. '
    'The confidence level indicates significant recognition errors.'
    '\n21. OCR Error from the User: it. To. To W. Boyd; Hayd, Carpenter'
    '   The JSON output is: '
    '     Corrected sentence: 1st. To Mr. Boyd, Paymaster.'
    '     Percentage of confidence: 61%'
    '     Justification: Corrected the misrecognition it. To. To W. Boyd; Hayd, '
    'Carpenter to 1st. To Mr. Boyd, Paymaster.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n22. OCR Error from the User: its to Colonel Stephen hashavight for Dorothy'
    '   The JSON output is: '
    '     Corrected sentence: As Colonel Stephen has brought'
    '     Percentage of confidence: 55%'
    '     Justification: Corrected its to Colonel Stephen hashavight for Dorothy to As Colonel Stephen has brought.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n23. OCR Error from the User: 6000n which I wrote for to pay off the Shop Shop,'
    '   The JSON output is: '
    '     Corrected sentence: 1000 which I wrote for to pay off the Troops,'
    '     Percentage of confidence: 79%'
    '     Justification: Corrected 6000n which I wrote for to pay off the Shop Shop to '
    '1000 which I wrote for to pay off the Troops.'
    'The relatively high confidence suggests good recognition accuracy for most parts.'
    '\n24. OCR Error from the User: it will serve a journey to Williams-'
    '   The JSON output is: '
    '     Corrected sentence: it will save you a journey to Williams-'
    '     Percentage of confidence: 85%'
    '     Justification: Corrected serve to save you.'
    'The relatively high confidence indicates good recognition accuracy.'
    '\n25. OCR Error from the User: lung at this time: but I think it think it absolutely'
    '   The JSON output is: '
    '     Corrected sentence: burg at this time: but I think it absolutely'
    '     Percentage of confidence: 79%'
    '     Justification: Corrected lung to burg and removed repeated words.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n26. OCR Error from the User: necgregary that you you should, after paying the'
    '   The JSON output is: '
    '     Corrected sentence: necessary that you should, after paying the'
    '     Percentage of confidence: 83%'
    '     Justification: Corrected necgregary to necessary and removed repeated words.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n27. OCR Error from the User: Kroops in Garrison, go into ahts, Augusta, to pay'
    '   The JSON output is: '
    '     Corrected sentence: Troops in Garrison, go into Augusta, to pay'
    '     Percentage of confidence: 85%'
    '     Justification: Corrected Kroops to Troops and ahts to Augusta.'
    'The confidence level indicates good recognition accuracy.'
    '\n28. OCR Error from the User: complete your can send the money'
    '   The JSON output is: '
    '     Corrected sentence: complete; unless you can send the money'
    '     Percentage of confidence: 77%'
    '     Justification: Corrected your to unless you.'
    'The confidence level indicates moderate recognition errors.'
    '\n29. OCR Error from the User: ly. No. Alexander, or some half'
    '   The JSON output is: '
    '     Corrected sentence: by Mr. Mc. Clenachan, or some safe hand'
    '     Percentage of confidence: 51%'
    '     Justification: Corrected ly. No. Alexander, or some half to by Mr. Mc. Clenachan, or some safe hand.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n30. OCR Error from the User: former herea. The Recruits about'
    '   The JSON output is: '
    '     Corrected sentence: from hence. The Recruits at Fort'
    '     Percentage of confidence: 66%'
    '     Justification: Corrected former herea to from hence and about to at Fort.'
    'The confidence level indicates some recognition errors.'
    '\n31. OCR Error from the User: tcher as private men men, here being no distinc-'
    '   The JSON output is: '
    '     Corrected sentence: tober as private men, there being no distinc-'
    '     Percentage of confidence: 85%'
    '     Justification: Corrected tcher as private men men to tober as private men.'
    'The confidence level indicates good recognition accuracy.'
    '\n32. OCR Error from the User: tion made between them and Sergeant,'
    '   The JSON output is: '
    '     Corrected sentence: tion made between them and Sergeants,'
    '     Percentage of confidence: 97%'
    '     Justification: Corrected Sergeant to Sergeants.'
    'The high confidence indicates excellent recognition accuracy.'
    '\n33. OCR Error from the User: none having her regularly.'
    '   The JSON output is: '
    '     Corrected sentence: none having yet been regularly appointed.'
    '     Percentage of confidence: 58%'
    '     Justification: Corrected none having her regularly to none having yet been regularly appointed.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n34. OCR Error from the User: donut of those he he hasard and any and how'
    '   The JSON output is: '
    '     Corrected sentence: count of those he has paid, and how -'
    '     Percentage of confidence: 60%'
    '     Justification: Corrected donut of those he he hasard and any and how '
    'to count of those he has paid, and how -.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n35. OCR Error from the User: Brianna, have received Sergeant Regents pay, it'
    '   The JSON output is: '
    '     Corrected sentence: If any have received Sergeants pay, it'
    '     Percentage of confidence: 72%'
    '     Justification: Corrected Brianna to If any and Regents to Sergeants.'
    'The confidence level indicates some recognition errors.'
    '\n36. OCR Error from the User: must be deposited next many months as also'
    '   The JSON output is: '
    '     Corrected sentence: must be deducted next payment: as also'
    '     Percentage of confidence: 76%'
    '     Justification: Corrected deposited to deducted and removed many months.'
    'The confidence level indicates some recognition errors.'
    '\n37. OCR Error from the User: instagram per month, from each non-commissions and'
    '   The JSON output is: '
    '     Corrected sentence: two - pence per month, from each non - commissioned'
    '     Percentage of confidence: 67%'
    '     Justification: Corrected instagram to two - pence and non-commissions and to non - commissioned.'
    'The confidence level indicates some recognition errors.'
    '\n38. OCR Error from the User: shes is to be paid to paid to the Trojan Quarterly. Have\'s'
    '   The JSON output is: '
    '     Corrected sentence: this is to be paid to the Surgeon Quarterly. There is'
    '     Percentage of confidence: 63%'
    '     Justification: Corrected shes is to this is, paid to paid to the Trojan to paid to the Surgeon, '
    'and Have\'s to There is.'
    'The confidence level indicates some recognition errors.'
    '\n39. OCR Error from the User: ness, to be paid to prior to the Drum. Major Major for tracking them.'
    '   The JSON output is: '
    '     Corrected sentence: mers, to be paid to the Drum - Major for teaching them,'
    '     Percentage of confidence: 71%'
    '     Justification: Corrected ness to mers, prior to the Drum to the Drum - Major, and tracking to teaching.'
    'The confidence level indicates some recognition errors.'
    '\n40. OCR Error from the User: llessu hereby Blanky Fahrenheit.'
    '   The JSON output is: '
    '     Corrected sentence: liver twenty Blankets.'
    '     Percentage of confidence: 36%'
    '     Justification: Corrected llessu hereby Blanky Fahrenheit to liver twenty Blankets.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n41. OCR Error from the User: December, with all the men you can'
    '   The JSON output is: '
    '     Corrected sentence: December, with all the men you can '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n42. OCR Error from the User: raise by that time.'
    '   The JSON output is: '
    '     Corrected sentence: raise by that time. '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n43. OCR Error from the User: Given Vc. at Fredericksburg,'
    '   The JSON output is: '
    '     Corrected sentence: Given Vc. at Fredericksburg, '
    '     Percentage of confidence: 32% '
    '     Justification: Corrected the misrecognition Given V.c.D. to Given Vc. at Fredericksburg.'
    'The low confidence indicates significant recognition errors.'
    '\n44. OCR Error from the User: November 1st. 1755.'
    '   The JSON output is: '
    '     Corrected sentence: November 1st. 1755. '
    '     Percentage of confidence: 84% '
    '     Justification: Corrected the misrecognition November 5. 1755. to November 1st. 1755.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n45. OCR Error from the User: G.W. Aid de camp.'
    '   The JSON output is: '
    '     Corrected sentence: G.W. Aid de camp. '
    '     Percentage of confidence: 76% '
    '     Justification: Corrected the misrecognition Mr. Aid de Camp. to G.W. Aid de camp.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n46. OCR Error from the User: N.B. Captain Joshua Lewis is allowed to'
    '   The JSON output is: '
    '     Corrected sentence: N.B. Captain Joshua Lewis is allowed to '
    '     Percentage of confidence: 82% '
    '     Justification: Corrected the misrecognition N.B. Captain Lewis is allowed to to N.B. Captain Joshua Lewis is allowed to.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n47. OCR Error from the User: the 1st. of December, to Rendezvous at Alex-'
    '   The JSON output is: '
    '     Corrected sentence: the 1st. of December, to Rendezvous at Alex- '
    '     Percentage of confidence: 91% '
    '     Justification: Corrected the misrecognition the I. of December, to Rendezvous at Alexa- to the 1st. of December, to Rendezvous at Alex-.'
    'The high confidence suggests accurate recognition.'
    '\n48. OCR Error from the User: andria.'
    '   The JSON output is: '
    '     Corrected sentence: andria. '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n49. OCR Error from the User: 1st. To Mr. Boyd, Paymaster.'
    '   The JSON output is: '
    '     Corrected sentence: 1st. To Mr. Boyd, Paymaster. '
    '     Percentage of confidence: 63% '
    '     Justification: Corrected the misrecognition t. To. To W. Boyd, Paymentster to 1st. To Mr. Boyd, Paymaster.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n50. OCR Error from the User: As Colonel Stephen has brought'
    '   The JSON output is: '
    '     Corrected sentence: As Colonel Stephen has brought '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n51. OCR Error from the User: [1000 which I wrote for to pay off the Troops,'
    '   The JSON output is: '
    '     Corrected sentence: 1000 which I wrote for to pay off the Troops, '
    '     Percentage of confidence: 96% '
    '     Justification: Corrected the misrecognition 6.000 which I wrote for to pay off the Troops, to 1000 which I wrote for to pay off the Troops.'
    'The high confidence suggests accurate recognition.'
    '\n52. OCR Error from the User: it will save you a journey to Williams-'
    '   The JSON output is: '
    '     Corrected sentence: it will save you a journey to Williams- '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n53. OCR Error from the User: burg at this time: but I think it absolutely'
    '   The JSON output is: '
    '     Corrected sentence: burg at this time: but I think it absolutely '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n54. OCR Error from the User: necessary that you should, after paying the'
    '   The JSON output is: '
    '     Corrected sentence: necessary that you should, after paying the '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n55. OCR Error from the User: Troops in Garrison, go into Augusta, to pay'
    '   The JSON output is: '
    '     Corrected sentence: Troops in Garrison, go into Augusta, to pay '
    '     Percentage of confidence: 95% '
    '     Justification: Corrected the misrecognition Drops in Garrison, go into Augusta, to pay to Troops in Garrison, go into Augusta, to pay.'
    'The high confidence suggests accurate recognition.'
    '\n56. OCR Error from the User: off Captain Hoggs Company, which is now'
    '   The JSON output is: '
    '     Corrected sentence: off Captain Hoggs Company, which is now '
    '     Percentage of confidence: 75% '
    '     Justification: Corrected the misrecognition of Captain Hoggih Hogg Company, whichehch is now to off Captain Hoggs Company, which is now.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n57. OCR Error from the User: complete; unless you can send the money'
    '   The JSON output is: '
    '     Corrected sentence: complete; unless you can send the money '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n58. OCR Error from the User: by Mr. Mc. Clenachan, or some safe hand'
    '   The JSON output is: '
    '     Corrected sentence: by Mr. Mc. Clenachan, or some safe hand '
    '     Percentage of confidence: 44% '
    '     Justification: Corrected the misrecognition by N. Remember, or the hand to by Mr. Mc. Clenachan, or some safe hand.'
    'The low confidence indicates significant recognition errors.'
    '\n59. OCR Error from the User: from hence. The Recruits at Fort'
    '   The JSON output is: '
    '     Corrected sentence: from hence. The Recruits at Fort '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n60. OCR Error from the User: Cumberland are all paid off to the 1st. of Oc-'
    '   The JSON output is: '
    '     Corrected sentence: Cumberland are all paid off to the 1st. of Oc- '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n61. OCR Error from the User: in Maryland or Pennsylvania'
    '   The JSON output is: '
    '     Corrected sentence: in Maryland, when they '
    '     Percentage of confidence: 57.89% '
    '     Justification: Corrected in Maryland or Pennsylvania to in Maryland, when they. '
    'The confidence level indicates significant recognition errors.'
    '\n62. OCR Error from the User: can not be had here. but with our money'
    '   The JSON output is: '
    '     Corrected sentence: can not be had be had here. But with our our money '
    '     Percentage of confidence: 76% '
    '     Justification: Corrected can not be had here. but with our money to can not be had be had here. But with our our money and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n63. OCR Error from the User: it is is impossible; our paper not passing there.'
    '   The JSON output is: '
    '     Corrected sentence: it is is impossible, our paper not passing thec. '
    '     Percentage of confidence: 93.88% '
    '     Justification: Corrected it is is impossible; our paper not passing there to it is is impossible, our paper not passing thec. '
    'The confidence level indicates good recognition accuracy.'
    '\n64. OCR Error from the User: The recruiting Service goes on ex-'
    '   The JSON output is: '
    '     Corrected sentence: The recruiting goes on- '
    '     Percentage of confidence: 67.65% '
    '     Justification: Corrected The recruiting Service goes on ex- to The recruiting goes on-. '
    'The confidence level indicates some recognition errors.'
    '\n65. OCR Error from the User: tremely slow. Yesterday being a day appointed'
    '   The JSON output is: '
    '     Corrected sentence: tremely slow. Yesterday being a day appointed '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n66. OCR Error from the User: for Rendezvousing at this place, there came in'
    '   The JSON output is: '
    '     Corrected sentence: foriendo using place in '
    '     Percentage of confidence: 43.48% '
    '     Justification: Corrected for Rendezvousing at this place, there came in to foriendo using place in. '
    'The low confidence indicates significant recognition errors.'
    '\n67. OCR Error from the User: ten Officers with twenty men only. If I had'
    '   The JSON output is: '
    '     Corrected sentence: tenuilles with twenty menty men only. If if I f had '
    '     Percentage of confidence: 64.71% '
    '     Justification: Corrected ten Officers with twenty men only. If I had to tenuilles with twenty menty men only. If if I f had and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n68. OCR Error from the User: any other than paper money, and you appro-'
    '   The JSON output is: '
    '     Corrected sentence: any other than paper money, and you appear- '
    '     Percentage of confidence: 93.02% '
    '     Justification: Corrected any other than paper money, and you appro- to any other than paper money, and you appear-. '
    'The confidence level indicates good recognition accuracy.'
    '\n69. OCR Error from the User: ved of it; I would send to Pennsylvania'
    '   The JSON output is: '
    '     Corrected sentence: ved of it; I would send to Pennsylvania '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n70. OCR Error from the User: and the Borders of Carolina: I am confident,'
    '   The JSON output is: '
    '     Corrected sentence: and the Borders of Carolina: I am confident, '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n71. OCR Error from the User: men might be had there. Your Honor'
    '   The JSON output is: '
    '     Corrected sentence: men might be had he had there. Youlton Kono '
    '     Percentage of confidence: 69.77% '
    '     Justification: Corrected men might be had there. Your Honor to men might be had he had there. Youlton Kono. '
    'The confidence level indicates some recognition errors.'
    '\n72. OCR Error from the User: never having given any particular Directi-'
    '   The JSON output is: '
    '     Corrected sentence: never having given any particular Directi- '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n73. OCR Error from the User: ons about the Provisions; I should be glad to'
    '   The JSON output is: '
    '     Corrected sentence: ons about the Provisions; Should be glad be glad to '
    '     Percentage of confidence: 78.43% '
    '     Justification: Corrected ons about the Provisions; I should be glad to to ons about the Provisions; Should be glad be glad to and removed repeated words. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n74. OCR Error from the User: know, whether you would have more laid in'
    '   The JSON output is: '
    '     Corrected sentence: know, whether you would have more laid in '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n75. OCR Error from the User: than what will serve for twelve hundred men;'
    '   The JSON output is: '
    '     Corrected sentence: that what will serve some for twelve hundred menden; '
    '     Percentage of confidence: 82.69% '
    '     Justification: Corrected than what will serve for twelve hundred men; to that what will serve some for twelve hundred menden;. '
    'The confidence level indicates good recognition accuracy.'
    '\n76. OCR Error from the User: that I may give orders accordingly.'
    '   The JSON output is: '
    '     Corrected sentence: that I may give Orders accordingly. '
    '     Percentage of confidence: 97.14% '
    '     Justification: Corrected that I may give orders accordingly. to that I may give Orders accordingly.. '
    'The confidence level indicates high recognition accuracy.'
    '\n77. OCR Error from the User: As I can not now conceive, that'
    '   The JSON output is: '
    '     Corrected sentence: As I can not now conceive, that '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n78. OCR Error from the User: any great danger can be apprehended at'
    '   The JSON output is: '
    '     Corrected sentence: any great danger canger can be approached at '
    '     Percentage of confidence: 72.73% '
    '     Justification: Corrected any great danger can be apprehended at to any great danger canger can be approached at. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n79. OCR Error from the User: Fort Cumberland this Winter; I am sensible,'
    '   The JSON output is: '
    '     Corrected sentence: Fort Cumberland this Winter; I am sensible, '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n80. OCR Error from the User: that my constant attendance there, can not'
    '   The JSON output is: '
    '     Corrected sentence: that my constant attend attendance there here, cannot '
    '     Percentage of confidence: 75.47% '
    '     Justification: Corrected that my constant attendance there, can not to that my constant attend attendance there here, cannot and removed repeated words. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n81. OCR Error from the User: can be got in these parts: those which Major'
    '   The JSON output is: '
    '     Corrected sentence: can be got in these parts: those which Major'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n82. OCR Error from the User: Carlyle and Dalton contracted to furnish,'
    '   The JSON output is: '
    '     Corrected sentence: Gaughyle and Dalton contracted to furnish to furnish,'
    '     Percentage of confidence: 71.7% '
    '     Justification: Corrected Carlyle to Gaughyle and removed repeated words. '
    'The confidence level indicates significant recognition errors.'
    '\n83. OCR Error from the User: we are disappointed off. Shoes and Stockings'
    '   The JSON output is: '
    '     Corrected sentence: we are disappointed off. Shoes and Stockings'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n84. OCR Error from the User: we have, and can get more if wanted, but'
    '   The JSON output is: '
    '     Corrected sentence: we have, and can get more if wanted, but'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n85. OCR Error from the User: nothing else. I should be glad your Honor'
    '   The JSON output is: '
    '     Corrected sentence: nothing else. I should be glad your Honor'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n86. OCR Error from the User: would direct what is to be done in these ca-'
    '   The JSON output is: '
    '     Corrected sentence: would direct what is to be done in these Ca-'
    '     Percentage of confidence: 97.73% '
    '     Justification: Corrected ca- to Ca-. '
    'The high confidence suggests accurate recognition.'
    '\n87. OCR Error from the User: ses; and that you would be kind enough to'
    '   The JSON output is: '
    '     Corrected sentence: ses; and that you would be kind enough to'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n88. OCR Error from the User: desire the Treasurer to send some part of the'
    '   The JSON output is: '
    '     Corrected sentence: Assie the Treasurer to send some sort some part of the'
    '     Percentage of confidence: 75.93% '
    '     Justification: Corrected desire to Assie and some part to some sort some part. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n89. OCR Error from the User: money in gold and silver: were this done, we'
    '   The JSON output is: '
    '     Corrected sentence: money in gold and and silver: were: were this done, we'
    '     Percentage of confidence: 81.48% '
    '     Justification: Corrected money in gold and silver to money in gold and and silver and removed repeated words. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n90. OCR Error from the User: might often get necessaries for the Regiment'
    '   The JSON output is: '
    '     Corrected sentence: might often get necessaries for the Regiment'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n91. OCR Error from the User: it is it impossible, our paper not passing there.'
    '   The JSON output is: '
    '     Corrected sentence: it is is impossible; our paper not passing there.'
    '     Percentage of confidence: 95.92% '
    '     Justification: Corrected is it to is is and added the semicolon. '
    'The high confidence suggests accurate recognition.'
    '\n92. OCR Error from the User: treme slow. Yesterday being being a day appointed'
    '   The JSON output is: '
    '     Corrected sentence: tremely slow. Yesterday being a day appointed'
    '     Percentage of confidence: 88.24% '
    '     Justification: Corrected treme slow to tremely slow and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n93. OCR Error from the User: there. Your Honor never having given any particular Directi-'
    '   The JSON output is: '
    '     Corrected sentence: there. Your Honor never having given any particular Directi-'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n94. OCR Error from the User: it; I would send to Pennsylvania'
    '   The JSON output is: '
    '     Corrected sentence: it; I would send to Pennsylvania'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n95. OCR Error from the User: and the Borders of Carolina: I am confident,'
    '   The JSON output is: '
    '     Corrected sentence: and the Borders of Carolina: I am confident,'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n96. OCR Error from the User: men might be had there. Your Honor'
    '   The JSON output is: '
    '     Corrected sentence: men might be had there. Your Honor'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n97. OCR Error from the User: never having given any particular Directi-'
    '   The JSON output is: '
    '     Corrected sentence: never having given any particular Directi-'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n98. OCR Error from the User: ons about the Provisions; I should be glad to'
    '   The JSON output is: '
    '     Corrected sentence: ons about the Provisions; I should be glad to'
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n99. OCR Error from the User: know you would have laid in'
    '   The JSON output is: '
    '     Corrected sentence: know, whether you would have more laid in'
    '     Percentage of confidence: 65.85% '
    '     Justification: Corrected know you to know, whether you. '
    'The confidence level indicates significant recognition errors.'
    '\n100. OCR Error from the User: that what will serve for twelve hundred men;'
    '   The JSON output is: '
    '     Corrected sentence: than what will serve for twelve hundred men;'
    '     Percentage of confidence: 97.73% '
    '     Justification: Corrected that what to than what. '
    'The high confidence suggests accurate recognition.'
)

EXAMPLES_F_60 = (
    '\n1. OCR Error from the User: to the Return made at - Food- Coun-'
    '   The JSON output is: '
    '     Corrected sentence: to the Return made me at Fort Cum- '
    '     Percentage of confidence: 74% '
    '     Justification: Corrected the misrecognition Food- Coun- to Fort Cum-. '
    'The confidence level indicates some recognition errors.'
    '\n2. OCR Error from the User: Ireland, October, - I have sent more'
    '   The JSON output is: '
    '     Corrected sentence: berland, October 26th. - I have sent more '
    '     Percentage of confidence: 78% '
    '     Justification: Corrected the misrecognition Ireland, October to October 26th. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n3. OCR Error from the User: Suits than are necessary, in case of options'
    '   The JSON output is: '
    '     Corrected sentence: Suits than are necessary, in case of getting '
    '     Percentage of confidence: 89% '
    '     Justification: Corrected the misrecognition options to getting. '
    'The high confidence suggests accurate recognition for most parts.'
    '\n4. OCR Error from the User: a Recruit to. You must do the best you'
    '   The JSON output is: '
    '     Corrected sentence: a Recruit Vc. You must do the best you '
    '     Percentage of confidence: 95% '
    '     Justification: Corrected the misrecognition to to Vc. '
    'The high confidence indicates good recognition accuracy.'
    '\n5. OCR Error from the User: can indoing the men, as the the Merchants.'
    '   The JSON output is: '
    '     Corrected sentence: can in Lodging the men, as the Barracks '
    '     Percentage of confidence: 64% '
    '     Justification: Corrected indoing the men to in Lodging the men and removed repeated words. '
    'The confidence level indicates moderate recognition errors.'
    '\n6. OCR Error from the User: in the Hot are full. It\'s Sergeant Will -'
    '   The JSON output is: '
    '     Corrected sentence: in the Fort are full. As Sergeant Wil- '
    '     Percentage of confidence: 83% '
    '     Justification: Corrected Hot to Fort and It\'s to As. '
    'The confidence level indicates good recognition accuracy.'
    '\n7. OCR Error from the User: ger is waiting the Return of the the Keegan\'s'
    '   The JSON output is: '
    '     Corrected sentence: per is waiting the return of the waggons '
    '     Percentage of confidence: 76% '
    '     Justification: Corrected ger to per and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n8. OCR Error from the User: from the Fort for fornecaries for Captain Captain'
    '   The JSON output is: '
    '     Corrected sentence: from the Fort, for necessaries for Captain '
    '     Percentage of confidence: 69% '
    '     Justification: Corrected fornecaries to necessaries and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n9. OCR Error from the User: Rogues Company, so soon as as they arrive,'
    '   The JSON output is: '
    '     Corrected sentence: Hoggs Company; so soon as they arrive, '
    '     Percentage of confidence: 83% '
    '     Justification: Corrected Rogues to Hoggs and removed repeated words. '
    'The confidence level indicates good recognition accuracy.'
    '\n10. OCR Error from the User: you are to see to see that he receives such'
    '   The JSON output is: '
    '     Corrected sentence: you are to see that he receives such '
    '     Percentage of confidence: 84% '
    '     Justification: Removed repeated words. '
    'The confidence level indicates good recognition accuracy.'
    '\n11. OCR Error from the User: things as he has been for, andathah'
    '   The JSON output is: '
    '     Corrected sentence: things as he has orders for, and dispatch '
    '     Percentage of confidence: 71% '
    '     Justification: Corrected been for to orders for and corrected andathah to dispatch. '
    'The confidence level indicates some recognition errors.'
    '\n12. OCR Error from the User: him immediately. If no no other Morses, debug.'
    '   The JSON output is: '
    '     Corrected sentence: him immediately. If no other Horses, belong- '
    '     Percentage of confidence: 80% '
    '     Justification: Corrected no no to no and Morses, debug to Horses, belong-. '
    'The confidence level indicates some recognition errors.'
    '\n13. OCR Error from the User: engagee the her heremen to remain in remain with'
    '   The JSON output is: '
    '     Corrected sentence: engage the herdsmen to remain with '
    '     Percentage of confidence: 67% '
    '     Justification: Corrected engagee the her heremen to engage the herdsmen and removed repeated words. '
    'The confidence level indicates some recognition errors.'
    '\n14. OCR Error from the User: injury, or from me me. Now must engage'
    '   The JSON output is: '
    '     Corrected sentence: missary or from me. You must engage '
    '     Percentage of confidence: 74% '
    '     Justification: Corrected injury, or from me me to missary or from me. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n15. OCR Error from the User: all the Coopers you can can to make 1sts.'
    '   The JSON output is: '
    '     Corrected sentence: all the Coopers you can to make Barrels '
    '     Percentage of confidence: 73% '
    '     Justification: Removed repeated words and corrected 1sts to Barrels. '
    'The confidence level indicates moderate recognition accuracy.'
    '\n16. OCR Error from the User: for packing the Beef, Beef, and if any of the'
    '   The JSON output is: '
    '     Corrected sentence: for packing the Beef; and if any of the '
    '     Percentage of confidence: 84% '
    '     Justification: Removed repeated words. '
    'The confidence level indicates good recognition accuracy.'
    '\n17. OCR Error from the User: enchiladaly to work. You are to receive'
    '   The JSON output is: '
    '     Corrected sentence: mediately to work. You are to receive '
    '     Percentage of confidence: 82% '
    '     Justification: Corrected enchiladaly to immediately. '
    'The confidence level indicates some recognition errors.'
    '\n18. OCR Error from the User: Roasts, short, which you are will deliver to'
    '   The JSON output is: '
    '     Corrected sentence: Goose - which you will deliver to '
    '     Percentage of confidence: 66% '
    '     Justification: Corrected Roasts, short, to Goose -. '
    'The confidence level indicates some recognition errors.'
    '\n19. OCR Error from the User: Captain thinking as you pege'
    '   The JSON output is: '
    '     Corrected sentence: Captain Ashby\'s Company as you pass '
    '     Percentage of confidence: 54% '
    '     Justification: Corrected thinking as you pege to Ashby\'s Company as you pass. '
    'The confidence level indicates significant recognition errors.'
    '\n20. OCR Error from the User: directing the Netherlands.'
    '   The JSON output is: '
    '     Corrected sentence: by, directing him to be particularly care- '
    '     Percentage of confidence: 48% '
    '     Justification: Corrected directing the Netherlands to directing him to be particularly care-. '
    'The confidence level indicates significant recognition errors.'
    '\n21. OCR Error from the User: it. To. To W. Boyd; Hayd, Carpenter'
    '   The JSON output is: '
    '     Corrected sentence: 1st. To Mr. Boyd, Paymaster.'
    '     Percentage of confidence: 61%'
    '     Justification: Corrected the misrecognition it. To. To W. Boyd; Hayd, '
    'Carpenter to 1st. To Mr. Boyd, Paymaster.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n22. OCR Error from the User: its to Colonel Stephen hashavight for Dorothy'
    '   The JSON output is: '
    '     Corrected sentence: As Colonel Stephen has brought'
    '     Percentage of confidence: 55%'
    '     Justification: Corrected its to Colonel Stephen hashavight for Dorothy to As Colonel Stephen has brought.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n23. OCR Error from the User: 6000n which I wrote for to pay off the Shop Shop,'
    '   The JSON output is: '
    '     Corrected sentence: 1000 which I wrote for to pay off the Troops,'
    '     Percentage of confidence: 79%'
    '     Justification: Corrected 6000n which I wrote for to pay off the Shop Shop to '
    '1000 which I wrote for to pay off the Troops.'
    'The relatively high confidence suggests good recognition accuracy for most parts.'
    '\n24. OCR Error from the User: it will serve a journey to Williams-'
    '   The JSON output is: '
    '     Corrected sentence: it will save you a journey to Williams-'
    '     Percentage of confidence: 85%'
    '     Justification: Corrected serve to save you.'
    'The relatively high confidence indicates good recognition accuracy.'
    '\n25. OCR Error from the User: lung at this time: but I think it think it absolutely'
    '   The JSON output is: '
    '     Corrected sentence: burg at this time: but I think it absolutely'
    '     Percentage of confidence: 79%'
    '     Justification: Corrected lung to burg and removed repeated words.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n26. OCR Error from the User: necgregary that you you should, after paying the'
    '   The JSON output is: '
    '     Corrected sentence: necessary that you should, after paying the'
    '     Percentage of confidence: 83%'
    '     Justification: Corrected necgregary to necessary and removed repeated words.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n27. OCR Error from the User: Kroops in Garrison, go into ahts, Augusta, to pay'
    '   The JSON output is: '
    '     Corrected sentence: Troops in Garrison, go into Augusta, to pay'
    '     Percentage of confidence: 85%'
    '     Justification: Corrected Kroops to Troops and ahts to Augusta.'
    'The confidence level indicates good recognition accuracy.'
    '\n28. OCR Error from the User: complete your can send the money'
    '   The JSON output is: '
    '     Corrected sentence: complete; unless you can send the money'
    '     Percentage of confidence: 77%'
    '     Justification: Corrected your to unless you.'
    'The confidence level indicates moderate recognition errors.'
    '\n29. OCR Error from the User: ly. No. Alexander, or some half'
    '   The JSON output is: '
    '     Corrected sentence: by Mr. Mc. Clenachan, or some safe hand'
    '     Percentage of confidence: 51%'
    '     Justification: Corrected ly. No. Alexander, or some half to by Mr. Mc. Clenachan, or some safe hand.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n30. OCR Error from the User: former herea. The Recruits about'
    '   The JSON output is: '
    '     Corrected sentence: from hence. The Recruits at Fort'
    '     Percentage of confidence: 66%'
    '     Justification: Corrected former herea to from hence and about to at Fort.'
    'The confidence level indicates some recognition errors.'
    '\n31. OCR Error from the User: tcher as private men men, here being no distinc-'
    '   The JSON output is: '
    '     Corrected sentence: tober as private men, there being no distinc-'
    '     Percentage of confidence: 85%'
    '     Justification: Corrected tcher as private men men to tober as private men.'
    'The confidence level indicates good recognition accuracy.'
    '\n32. OCR Error from the User: tion made between them and Sergeant,'
    '   The JSON output is: '
    '     Corrected sentence: tion made between them and Sergeants,'
    '     Percentage of confidence: 97%'
    '     Justification: Corrected Sergeant to Sergeants.'
    'The high confidence indicates excellent recognition accuracy.'
    '\n33. OCR Error from the User: none having her regularly.'
    '   The JSON output is: '
    '     Corrected sentence: none having yet been regularly appointed.'
    '     Percentage of confidence: 58%'
    '     Justification: Corrected none having her regularly to none having yet been regularly appointed.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n34. OCR Error from the User: donut of those he he hasard and any and how'
    '   The JSON output is: '
    '     Corrected sentence: count of those he has paid, and how -'
    '     Percentage of confidence: 60%'
    '     Justification: Corrected donut of those he he hasard and any and how '
    'to count of those he has paid, and how -.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n35. OCR Error from the User: Brianna, have received Sergeant Regents pay, it'
    '   The JSON output is: '
    '     Corrected sentence: If any have received Sergeants pay, it'
    '     Percentage of confidence: 72%'
    '     Justification: Corrected Brianna to If any and Regents to Sergeants.'
    'The confidence level indicates some recognition errors.'
    '\n36. OCR Error from the User: must be deposited next many months as also'
    '   The JSON output is: '
    '     Corrected sentence: must be deducted next payment: as also'
    '     Percentage of confidence: 76%'
    '     Justification: Corrected deposited to deducted and removed many months.'
    'The confidence level indicates some recognition errors.'
    '\n37. OCR Error from the User: instagram per month, from each non-commissions and'
    '   The JSON output is: '
    '     Corrected sentence: two - pence per month, from each non - commissioned'
    '     Percentage of confidence: 67%'
    '     Justification: Corrected instagram to two - pence and non-commissions and to non - commissioned.'
    'The confidence level indicates some recognition errors.'
    '\n38. OCR Error from the User: shes is to be paid to paid to the Trojan Quarterly. Have\'s'
    '   The JSON output is: '
    '     Corrected sentence: this is to be paid to the Surgeon Quarterly. There is'
    '     Percentage of confidence: 63%'
    '     Justification: Corrected shes is to this is, paid to paid to the Trojan to paid to the Surgeon, '
    'and Have\'s to There is.'
    'The confidence level indicates some recognition errors.'
    '\n39. OCR Error from the User: ness, to be paid to prior to the Drum. Major Major for tracking them.'
    '   The JSON output is: '
    '     Corrected sentence: mers, to be paid to the Drum - Major for teaching them,'
    '     Percentage of confidence: 71%'
    '     Justification: Corrected ness to mers, prior to the Drum to the Drum - Major, and tracking to teaching.'
    'The confidence level indicates some recognition errors.'
    '\n40. OCR Error from the User: llessu hereby Blanky Fahrenheit.'
    '   The JSON output is: '
    '     Corrected sentence: liver twenty Blankets.'
    '     Percentage of confidence: 36%'
    '     Justification: Corrected llessu hereby Blanky Fahrenheit to liver twenty Blankets.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n41. OCR Error from the User: December, with all the men you can'
    '   The JSON output is: '
    '     Corrected sentence: December, with all the men you can '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n42. OCR Error from the User: raise by that time.'
    '   The JSON output is: '
    '     Corrected sentence: raise by that time. '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n43. OCR Error from the User: Given Vc. at Fredericksburg,'
    '   The JSON output is: '
    '     Corrected sentence: Given Vc. at Fredericksburg, '
    '     Percentage of confidence: 32% '
    '     Justification: Corrected the misrecognition Given V.c.D. to Given Vc. at Fredericksburg.'
    'The low confidence indicates significant recognition errors.'
    '\n44. OCR Error from the User: November 1st. 1755.'
    '   The JSON output is: '
    '     Corrected sentence: November 1st. 1755. '
    '     Percentage of confidence: 84% '
    '     Justification: Corrected the misrecognition November 5. 1755. to November 1st. 1755.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n45. OCR Error from the User: G.W. Aid de camp.'
    '   The JSON output is: '
    '     Corrected sentence: G.W. Aid de camp. '
    '     Percentage of confidence: 76% '
    '     Justification: Corrected the misrecognition Mr. Aid de Camp. to G.W. Aid de camp.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n46. OCR Error from the User: N.B. Captain Joshua Lewis is allowed to'
    '   The JSON output is: '
    '     Corrected sentence: N.B. Captain Joshua Lewis is allowed to '
    '     Percentage of confidence: 82% '
    '     Justification: Corrected the misrecognition N.B. Captain Lewis is allowed to to N.B. Captain Joshua Lewis is allowed to.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n47. OCR Error from the User: the 1st. of December, to Rendezvous at Alex-'
    '   The JSON output is: '
    '     Corrected sentence: the 1st. of December, to Rendezvous at Alex- '
    '     Percentage of confidence: 91% '
    '     Justification: Corrected the misrecognition the I. of December, to Rendezvous at Alexa- to the 1st. of December, to Rendezvous at Alex-.'
    'The high confidence suggests accurate recognition.'
    '\n48. OCR Error from the User: andria.'
    '   The JSON output is: '
    '     Corrected sentence: andria. '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n49. OCR Error from the User: 1st. To Mr. Boyd, Paymaster.'
    '   The JSON output is: '
    '     Corrected sentence: 1st. To Mr. Boyd, Paymaster. '
    '     Percentage of confidence: 63% '
    '     Justification: Corrected the misrecognition t. To. To W. Boyd, Paymentster to 1st. To Mr. Boyd, Paymaster.'
    'The relatively lower confidence indicates significant recognition errors.'
    '\n50. OCR Error from the User: As Colonel Stephen has brought'
    '   The JSON output is: '
    '     Corrected sentence: As Colonel Stephen has brought '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n51. OCR Error from the User: [1000 which I wrote for to pay off the Troops,'
    '   The JSON output is: '
    '     Corrected sentence: 1000 which I wrote for to pay off the Troops, '
    '     Percentage of confidence: 96% '
    '     Justification: Corrected the misrecognition 6.000 which I wrote for to pay off the Troops, to 1000 which I wrote for to pay off the Troops.'
    'The high confidence suggests accurate recognition.'
    '\n52. OCR Error from the User: it will save you a journey to Williams-'
    '   The JSON output is: '
    '     Corrected sentence: it will save you a journey to Williams- '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n53. OCR Error from the User: burg at this time: but I think it absolutely'
    '   The JSON output is: '
    '     Corrected sentence: burg at this time: but I think it absolutely '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n54. OCR Error from the User: necessary that you should, after paying the'
    '   The JSON output is: '
    '     Corrected sentence: necessary that you should, after paying the '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n55. OCR Error from the User: Troops in Garrison, go into Augusta, to pay'
    '   The JSON output is: '
    '     Corrected sentence: Troops in Garrison, go into Augusta, to pay '
    '     Percentage of confidence: 95% '
    '     Justification: Corrected the misrecognition Drops in Garrison, go into Augusta, to pay to Troops in Garrison, go into Augusta, to pay.'
    'The high confidence suggests accurate recognition.'
    '\n56. OCR Error from the User: off Captain Hoggs Company, which is now'
    '   The JSON output is: '
    '     Corrected sentence: off Captain Hoggs Company, which is now '
    '     Percentage of confidence: 75% '
    '     Justification: Corrected the misrecognition of Captain Hoggih Hogg Company, whichehch is now to off Captain Hoggs Company, which is now.'
    'The confidence level indicates moderate recognition accuracy.'
    '\n57. OCR Error from the User: complete; unless you can send the money'
    '   The JSON output is: '
    '     Corrected sentence: complete; unless you can send the money '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n58. OCR Error from the User: by Mr. Mc. Clenachan, or some safe hand'
    '   The JSON output is: '
    '     Corrected sentence: by Mr. Mc. Clenachan, or some safe hand '
    '     Percentage of confidence: 44% '
    '     Justification: Corrected the misrecognition by N. Remember, or the hand to by Mr. Mc. Clenachan, or some safe hand.'
    'The low confidence indicates significant recognition errors.'
    '\n59. OCR Error from the User: from hence. The Recruits at Fort'
    '   The JSON output is: '
    '     Corrected sentence: from hence. The Recruits at Fort '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
    '\n60. OCR Error from the User: Cumberland are all paid off to the 1st. of Oc-'
    '   The JSON output is: '
    '     Corrected sentence: Cumberland are all paid off to the 1st. of Oc- '
    '     Percentage of confidence: 100% '
    '     Justification: The sentence is correctly recognized. '
    'The high confidence suggests accurate recognition.'
)

EXAMPLES_F_5 = (
    '\n1. OCR Error from the User: 20%. Letterss orders and Instructions and Instructions Decembers December 1755.'
    '   The JSON output is: '
    '     Corrected sentence: 308. Letters Orders and Instructions December 1755. '
    '     Percentage of confidence: 61% '
    '     Justification: Corrected the misrecognition 20% to 308 and removed repeated phrases. '
    'The relatively lower confidence indicates significant recognition errors.'
    '\n2. OCR Error from the User: remain here until the arrival of the visit with '
    '   The JSON output is: '
    '     Corrected sentence: remain here until the arrival of the vessel with '
    '     Percentage of confidence: 92% '
    '     Justification: Corrected the misrecognition visit to vessel. The high confidence '
    'indicates most of the text was correctly recognized.'
    '\n3. OCR Error from the User: the Flours, Vc. and to be under the same directions '
    '   The JSON output is: '
    '     Corrected sentence: the Stores, Vc. and to be under the same directions '
    '     Percentage of confidence: 92% '
    '     Justification: Corrected the misrecognition Flours to Stores. The high confidence suggests '
    'accurate recognition for most parts.'
)

EXAMPLES_F_4 = (
    '\n1. OCR Error from the User: 20%. Letterss orders and Instructions and Instructions Decembers December 1755.'
    '   The JSON output is: '
    '     Original sentence: 20%. Letterss orders and Instructions and Instructions Decembers December 1755.'
    '     Corrected sentence: 308. Letters Orders and Instructions December 1755. '
    '     Confidence (%): 61 '
    '     Justification: Corrected the misrecognition 20% to 308 and removed repeated phrases. '
    'The relatively lower confidence indicates significant recognition errors.'
)

EXAMPLES_F_10 = (
    '\n1. OCR Error from the User: only for the publick use, unless by particu-'
    '   The JSON output is: '
    '     Original text line: only hor the plbrick fse, unless by particu-'
    '     Corrected text line: only for the publick use, unless by particu-'
    '     Confidence (%): 92 '
    '     Justification: Corrected "hor" to "for", "plbrick" to "publick", "fse" to "use", and removed unnecessary '
    'spacing. Ensured the restored text adheres to the original meaning and context.'

    '\n2. OCR Error from the User: down a Barrel of Flints with the Arms, to'
    '   The JSON output is: '
    '     Original text line: down h Barrel of Fuints with the mr1s, to'
    '     Corrected text line: down a Barrel of Flints with the Arms, to'
    '     Confidence (%): 86 '
    '     Justification: Corrected "h" to "a", "Fuints" to "Flints", "mr1s" to "Arms", ensuring accuracy in the '
    'historical context and terminology used.'

    '\n3. OCR Error from the User: Plantation of Charles Sellars - the rest to Captain'
    '   The JSON output is: '
    '     Original text line: PkanIaWion on Charles Sellars - the rest to haptain'
    '     Corrected text line: Plantation of Charles Sellars - the rest to Captain'
    '     Confidence (%): 55 '
    '     Justification: Corrected "PkanIaWion" to "Plantation", "on" to "of", "haptain" to "Captain", ensuring proper '
    'names and titles are accurately represented.'

    '\n4. OCR Error from the User: The Officers who came down'
    '   The JSON output is: '
    '     Original text line: The Ofbicers who caTe down'
    '     Corrected text line: The Officers who came down'
    '     Confidence (%): 98 '
    '     Justification: Corrected "Ofbicers" to "Officers", "caTe" to "came", ensuring clarity and proper grammatical'
    ' structure.'

    '\n5. OCR Error from the User: to Repair to Captain Hoggs Company with eight'
    '   The JSON output is: '
    '     Original text line: to RepaiO to Captain Hoggs nompany wi h Dight'
    '     Corrected text line: to Repair to Captain Hoggs Company with eight'
    '     Confidence (%): 83 '
    '     Justification: Corrected "RepaiO" to "Repair", "nompany" to "Company", "wi h" to "with", "Dight" to "eight", '
    'ensuring accuracy in spelling and context.'

    '\n6. OCR Error from the User: one Sergeant, one Corporal, one Drummer,'
    '   The JSON output is: '
    '     Original text line: omeh ergeant, one Corporal, one DrPmmer,'
    '     Corrected text line: one Sergeant, one Corporal, one Drummer,'
    '     Confidence (%): 79 '
    '     Justification: Corrected "omeh" to "one", "ergeant" to "Sergeant", "DrPmmer" to "Drummer", ensuring correct'
    ' identification of ranks and roles.'

    '\n7. OCR Error from the User: without a Surgeon, if you will do that duty, an'
    '   The JSON output is: '
    '     Original text line: without a SurgJon, if you lilt do that dutn, an'
    '     Corrected text line: without a Surgeon, if you will do that duty, an'
    '     Confidence (%): 72 '
    '     Justification: Corrected "SurgJon" to "Surgeon", "lilt" to "will", "dutn" to "duty", ensuring the sentence '
    'is clear and medically accurate.'

    '\n8. OCR Error from the User: halt there until he joins, in order to escort the'
    '   The JSON output is: '
    '     Original text line: halt there untyl he joins,sin order Wo escort thh'
    '     Corrected text line: halt there until he joins, in order to escort the'
    '     Confidence (%): 88 '
    '     Justification: Corrected "untyl" to "until", "joins,sin" to "joins, in", "Wo" to "to", "thh" to "the", '
    'ensuring the text maintains its intended meaning and readability.'

    '\n9. OCR Error from the User: are to supply him from the'
    '   The JSON output is: '
    '     Original text line: are yo supply him from tpe'
    '     Corrected text line: are to supply him from the'
    '     Confidence (%): 79 '
    '     Justification: Corrected "yo" to "to", "tpe" to "the", ensuring the sentence is grammatically correct '
    'and clear.'

    '\n10. OCR Error from the User: march them immediately to Fort Cumberland;'
    '   The JSON output is: '
    '     Original text line: march -hemUimmediately to Foro Cumbe,land;'
    '     Corrected text line: march them immediately to Fort Cumberland;'
    '     Confidence (%): 81'
    '     Justification: Corrected "-hemUimmediately" to "them immediately", "Foro" to "Fort", "Cumbe,land" to '
    '"Cumberland", ensuring the sentence accurately reflects the intended instruction.'
)

EXAMPLES_F_3 = (
    "\n1. OCR Error from the User: The quick brown fox jumpd over the lazy dog."
    " The JSON output is: "
    " Corrected sentence: The quick brown fox jumped over the lazy dog."
    " Percentage of confidence: 98%"
    " Justification: Corrected jumpd to jumped."
    "The confidence level indicates high recognition accuracy."
    "\n2. OCR Error from the User: She walkd to ye market to buy apples."
    " The JSON output is: "
    " Corrected sentence: She walked to the market to buy apples."
    " Percentage of confidence: 95%"
    " Justification: Corrected walkd to walked and ye to the."
    "The confidence level indicates high recognition accuracy."
    "\n3. OCR Error from the User: The king's speech was praise"
    " The JSON output is: "
    " Corrected sentence: The king's speech was praise"
    " Percentage of confidence: 80%"
    " Justification: Left the sentence as is due to cut-out words."
    "The confidence level indicates moderate recognition accuracy."
    "\n4. OCR Error from the User: The house was built in 1776."
    " The JSON output is: "
    " Corrected sentence: The house was built in 1776."
    " Percentage of confidence: 95%"
    " Justification: Sentence is clear and accurate."
    "The confidence level indicates high recognition accuracy."
    "\n5. OCR Error from the User: He plucked the rose and handed it to hare."
    " The JSON output is: "
    " Corrected sentence: He plucked the rose and handed it to her."
    " Percentage of confidence: 92%"
    " Justification: Corrected hare to her based on context."
    "The confidence level indicates high recognition accuracy."
    "\n6. OCR Error from the User: The cattt sat on the mat."
    " The JSON output is: "
    " Corrected sentence: The cat sat on the mat."
    " Percentage of confidence: 96%"
    " Justification: Corrected cattt to cat."
    "The confidence level indicates high recognition accuracy."
)

OCR_ERRORS = (
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

)


EXAMPLES_V2_F_10 = (
    f"\n1. The text line contains OCR errors?: 'Dear Sir, Oxon, Sept. 17, 1734.'"
    f"The output is:"
    f"   Response: No"
    f"   Explanation: The text line 'Dear Sir, Oxon, Sept. 17, 1734.' is correct and does not contain any OCR errors."

    f"\n2. The text line contains OCR errors?: 'I Had the favour of your letter last Friday, which brought'"
    f"The output is:"
    f"   Response: No"
    f"   Explanation: The text line 'I Had the favour of your letter last Friday, which brought' is correct and does "
    f"not contain any OCR errors."

    f"\n3. The text line contains OCR errors?: 'meeting; with a candid reception, from all those persons to'"
    f"The output is:"
    f"   Response: No"
    f"   Explanation: The text line 'meeting; with a candid reception, from all those persons to' is correct "
    f"and does not contain any OCR errors."

    f"\n4. The text line contains OCR errors?: 'gethAr wier theSpucWSdasir:d awcouetAof sour approving ehe'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'gethAr' instead of 'gether', 'wier' instead of 'with', "
    f"'theSpucWSdasir:d' instead of 'the much-desired', 'awcouetAof' instead of 'account of', 'sour' instead of "
    f"'your', and 'ehe' instead of 'the'. These were corrected to 'gether with the much-desired account of your "
    f"approving the'."

    f"\n5. The text line contains OCR errors?: 'wc7emep inclosed iy fy lbst. Indee.t I did got doebxxofnhts'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'wc7emep' instead of 'scheme', 'inclosed iy' instead of 'inclosed in', "
    f"'fy lbst.' instead of 'my last.', 'Indee.t' instead of 'Indeed,', 'I did got' instead of 'I did not', "
    f"and 'doebxxofnhts' instead of 'doubt of its'. These were corrected to 'scheme, inclosed in my last. Indeed, "
    f"I did not doubt of its'."

    f"\n6. The text line contains OCR errors?: 'meeting;hwith a ca:di  ec1eptiovt from bly 7hdse petsonf to'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'meeting;hwith' instead of 'meeting; with', 'a ca:di' instead of 'a "
    f"candid', 'ec1eptiovt' instead of 'reception,', 'from bly 7hdse' instead of 'from all those', and 'petsonf' "
    f"instead of 'persons to'. These were corrected to 'meeting; with a candid reception, from all those persons to'."

    f"\n7. The text line contains OCR errors?: 'whlT i, wasarecommnndodgSTWeir knort concere for relie'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'whlT i,' instead of 'whom it', 'wasarecommnndodgSTWeir' instead of 'was "
    f"recommended. Their', 'knort' instead of 'known', 'concere' instead of 'concern', and 'relie' instead of "
    f"'reli-'. These were corrected to 'whom it was recommended. Their known concern for reli-'."

    f"\n8. The text line contains OCR errors?: 'gion. gkvocg mi sidficient assuiancT, taatAnythihg 1a: be un-'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'gion. gkvocg' instead of 'gion, giving', 'mi sidficient' instead of 'me "
    f"sufficient', 'assuiancT,' instead of 'assurance,', 'taatAnythihg 1a: be un-' instead of 'that nothing can be "
    f"un-'. These were corrected to 'gion, giving me sufficient assurance, that nothing can be un-'."

    f"\n9. The text line contains OCR errors?: 'aVcept:blA sA thed, whech apy way tgnded to proaofu th Sr'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'aVcept:blA sA thed,' instead of 'acceptable to them,', 'whech apy way "
    f"tgnded' instead of 'which any way tended', and 'proaofu th Sr' instead of 'to promote their'. These were "
    f"corrected to 'acceptable to them, which any way tended to promote their'."

    f"\n10. The text line contains OCR errors?: 'improvemext Wn thD divito life.DIhamuIc b; lonfvssRp, indeed,'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'improvemext' instead of 'improvement', 'Wn thD' instead of 'in the', "
    f"'divito' instead of 'divine', 'life.DIhamuIc' instead of 'life. It must', 'b; lonfvssRp,' instead of 'be "
    f"confessed,', and 'indeed,' remains the same. These were corrected to 'improvement in the divine life. It must "
    f"be confessed, indeed,'."
)


EXAMPLES_F_10 = (
    '\n1. OCR Error from the User: Dea, Cmr, Oxon, Seprs  7,u1734.'
    '   The JSON output is: '
    '     Original text line: Dea, Cmr, Oxon, Seprs  7,u1734.'
    '     Corrected text line: Dear Sir, Oxon, Sept. 17, 1734.'
    '     Confidence (%): 95 '
    'Justification: Corrected the misrecognized characters "Dea, Cmr" to "Dear Sir", "Seprs  7,u1734" to "Sept. 17, '
    '1734".'

    '\n2. OCR Error from the User: I Had the fnvourcofuyot1Isetter las7sFoi.ay,7wpich bruught'
    '   The JSON output is: '
    '     Original text line: I Had the fnvourcofuyot1Isetter las7sFoi.ay,7wpich bruught'
    '     Corrected text line: I Had the favour of your letter last Friday, which brought'
    '     Confidence (%): 94 '
    '     Justification: Corrected the substitutions "fnvourcofuyot1Isetter las7sFoi.ay,7wpich bruught" to "favour of your letter last Friday, which brought". '

    '\n3. OCR Error from the User: eemxhe agweeabtc Oews of wour ony M.sz H,\'s relfare, tG-'
    '   The JSON output is: '
    '     Original text line: eemxhe agweeabtc Oews of wour ony M.sz H,\'s relfare, tG-'
    '     Corrected text line: me the agreeable news of your and Mrs, H,\'s welfare, to-'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "eemxhe agweeabtc Oews of wour ony M.sz H,\'s relfare, tG-" to "me the agreeable news of your and Mrs, H,\'s welfare, to-". '

    '\n4. OCR Error from the User: getc,r iith tae much-deFvred atcountBof yodra,ptrovivg tce'
    '   The JSON output is: '
    '     Original text line: getc,r iith tae much-deFvred atcountBof yodra,ptrovivg tce'
    '     Corrected text line: gether with the much-desired account of your approving the'
    '     Confidence (%): 91 '
    '     Justification: Corrected the substitutions "getc,r iith tae much-deFvred atcountBof yodra,ptrovivg tce" to "gether with the much-desired account of your approving the". '

    '\n5. OCR Error from the User: schemh, incleeeddTn my laut. Indee., I did not dJuetVof:uim'
    '   The JSON output is: '
    '     Original text line: schemh, incleeeddTn my laut. Indee., I did not dJuetVof:uim'
    '     Corrected text line: scheme, inclosed in my last. Indeed, I did not doubt of its'
    '     Confidence (%): 89 '
    '     Justification: Corrected the substitutions "schemh, incleeeddTn my laut. Indee., I did not dJuetVof:uim" to "scheme, inclosed in my last. Indeed, I did not doubt of its". '

    '\n6. OCR Error from the User: mpdtiag;;with l cVndid eeception,:from all thosbOperpo;sIgo'
    '   The JSON output is: '
    '     Original text line: mpdtiag;;with l cVndid eeception,:from all thosbOperpo;sIgo'
    '     Corrected text line: meeting; with a candid reception, from all those persons to'
    '     Confidence (%): 97 '
    '     Justification: Corrected the substitutions "mpdtiag;;with l cVndid eeception,:from all thosbOperpo;sIgo" to "meeting; with a candid reception, from all those persons to". '

    '\n7. OCR Error from the User: whom ibnias,rdcomyun ed.ITheir k:awn concern foF relw-'
    '   The JSON output is: '
    '     Original text line: whom ibnias,rdcomyun ed.ITheir k:awn concern foF relw-'
    '     Corrected text line: whom it was recommended. Their known concern for reli-'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "whom ibnias,rdcomyun ed.ITheir k:awn concern foF relw-" to "whom it was recommended. Their known concern for reli-". '

    '\n8. OCR Error from the User: gioa, Vguing te sufficientTassuranba, toat nothiee dSn e, un-'
    '   The JSON output is: '
    '     Original text line: gioa, Vguing te sufficientTassuranba, toat nothiee dSn e, un-'
    '     Corrected text line: gion, giving me sufficient assurance, that nothing can be un-'
    '     Confidence (%): 72 '
    '     Justification: Corrected the substitutions "gioa, Vguing te sufficientTassuranba, toat nothiee dSn e, un-" to "gion, giving me sufficient assurance, that nothing can be un-". '

    '\n9. OCR Error from the User: alceptable tg tnemr yhWch asy wa  teJhed to Gromotv xhmir'
    '   The JSON output is: '
    '     Original text line: alceptable tg tnemr yhWch asy wa  teJhed to Gromotv xhmir'
    '     Corrected text line: acceptable to them, which any way tended to promote their'
    '     Confidence (%): 61 '
    '     Justification: Corrected the substitutions "alceptable tg tnemr yhWch asy wa  teJhed to Gromotv xhmir" to "acceptable to them, which any way tended to promote their". '

    '\n10. OCR Error from the User: improvzmentbin t.:vdivioe whOe..It oust be uorfessed, indJedC'
    '   The JSON output is: '
    '     Original text line: improvzmentbin t.:vdivioe whOe..It oust be uorfessed, indJedC'
    '     Corrected text line: improvement in the divine life. It must be confessed, indeed,'
    '     Confidence (%): 45 '
    '     Justification: Corrected the substitutions "improvzmentbin t.:vdivioe whOe..It oust be uorfessed, indJedC" to "improvement in the divine life. It must be confessed, indeed,". '
)

EXAMPLES_MISRECOGNIZED_ERRORS = (
    f"\n1. Does the given text line contain misrecognized characters or number errors? The text line is: 'Dear Sir, "
    f"Oxon, Sept. 17, 1734.'"
    f"The output is:"
    f"   Response: No"

    f"\n2. Does the given text line contain misrecognized characters or number errors? The text line is: 'IaNid "
    f"theDf:coua of your l.tterylast Fridih, whRchvDrought'"
    f"The output is:"
    f"   Response: Yes"
)

EXAMPLES_CORRECTOR_MISRECOGNIZED_ERRORS = (
    f"\n1. Given the text line: IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought "
    f"The output is:"
    f"  Corrected text line is: I Had the favour of your letter last Friday, which brought"
)

EXAMPLES_INCORRECT_ABBREVIATIONS = (
    f"\n1. Does the given text line contain incorrect abbreviations? The text line is: 'Dr. James and Mr. Smith.' "
    f"The output is:"
    f"   Response: No"

    f"\n2. Does the given text line contain incorrect abbreviations? The text line is: 'D. James and Mr. S.' "
    f"The output is:"
    f"   Response: Yes"
)

EXAMPLES_CORRECTOR_INCORRECT_ABBREVIATIONS = (
    f"\n1. Given the text line: D. James and Mr. S. "
    f"The output is:"
    f"  Corrected text line is: Dr. James and Mr. Smith"
)


EXAMPLES_O_V2_10 = (
    f"\n1. The text line contains OCR errors?: 'Dear Sir, Oxon, Sept. 17, 1734.'"
    f"The output is:"
    f"   Response: No"
    f"   Explanation: The text line 'Dear Sir, Oxon, Sept. 17, 1734.' is correct and does not contain any OCR errors."

    f"\n2. The text line contains OCR errors?: 'I Had the favour of your letter last Friday, which brought'"
    f"The output is:"
    f"   Response: No"
    f"   Explanation: The text line 'I Had the favour of your letter last Friday, which brought' is correct and does "
    f"not contain any OCR errors."

    f"\n3. The text line contains OCR errors?: 'meeting; with a candid reception, from all those persons to'"
    f"The output is:"
    f"   Response: No"
    f"   Explanation: The text line 'meeting; with a candid reception, from all those persons to' is correct and does "
    f"not contain any OCR errors."

    f"\n4. The text line contains OCR errors?: 'Dekr Sir,,Oxou, ST-t.v1e, 1734.'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'Dekr' instead of 'Dear', 'Sir,,' instead of 'Sir,', 'Oxou,' instead of 'Oxon,', 'ST-t.v1e,' instead of 'Sept. 17,'."

    f"\n5. The text line contains OCR errors?: 'IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'IaNid' instead of 'I Had', 'theDf:coua' instead of 'the favour', 'l.tterylast' instead of 'letter last', 'Fridih,' instead of 'Friday,', 'whRchvDrought' instead of 'which brought'."

    f"\n6. The text line contains OCR errors?: 'me S5e Dgreeable newd Sl yourtan, Mrs, N,\'m oelftre, dt-'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'me S5e Dgreeable newd' instead of 'me the agreeable news', 'Sl yourtan,' instead of 'of your and', 'Mrs, N,\'m oelftre,' instead of 'Mrs, H,\'s welfare,'."

    f"\n7. The text line contains OCR errors?: 'gethAr wier theSpucWSdasir:d awcouetAof sour approving ehe'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'gethAr wier' instead of 'gether with', 'theSpucWSdasir:d' instead of 'the much-desired', 'awcouetAof' instead of 'account of', 'sour approving' instead of 'your approving', 'ehe' instead of 'the'."

    f"\n8. The text line contains OCR errors?: 'wc7emep inclosed iy fy lbst. Indee.t I did got doebxxofnhts'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'wc7emep' instead of 'scheme,', 'inclosed iy' instead of 'inclosed in', 'fy lbst.' instead of 'my last.', 'Indee.t' instead of 'Indeed,', 'I did got doebxxofnhts' instead of 'I did not doubt of its'."

    f"\n9. The text line contains OCR errors?: 'meeting;hwith a ca:di  ec1eptiovt from bly 7hdse petsonf to'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'meeting;hwith' instead of 'meeting; with', 'a ca:di  ec1eptiovt' instead of 'a candid reception,', 'from bly 7hdse' instead of 'from all those', 'petsonf' instead of 'persons'."

    f"\n10. The text line contains OCR errors?: 'whlT i, wasarecommnndodgSTWeir knort concere for relie'"
    f"The output is:"
    f"   Response: Yes"
    f"   Explanation: The OCR errors include 'whlT i,' instead of 'whom it', 'wasarecommnndodg' instead of 'was recommended.', 'STWeir knort' instead of 'Their known', 'concere' instead of 'concern', 'relie' instead of 'reli-'."
)


EXAMPLES_O_10 = (
    '\n1. OCR Error from the User: Dekr Sir,,Oxou, ST-t.v1e, 1734.'
    '   The JSON output is: '
    '     Original text line: Dekr Sir,,Oxou, ST-t.v1e, 1734.'
    '     Corrected text line: Dear Sir, Oxon, Sept. 17, 1734.'
    '     Confidence (%): 95 '
    '     Justification: Corrected the misrecognized characters "Dekr" to "Dear", "Sir,," to "Sir,", "Oxou," to "Oxon,", "ST-t.v1e," to "Sept. 17,".'

    '\n2. OCR Error from the User: IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought'
    '   The JSON output is: '
    '     Original text line: IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought'
    '     Corrected text line: I Had the favour of your letter last Friday, which brought'
    '     Confidence (%): 94 '
    '     Justification: Corrected the substitutions "IaNid" to "I Had", "theDf:coua" to "the favour", "l.tterylast" to "letter last", "Fridih," to "Friday,", "whRchvDrought" to "which brought".'

    '\n3. OCR Error from the User: me S5e Dgreeable newd Sl yourtan, Mrs, N,\'m oelftre, dt-'
    '   The JSON output is: '
    '     Original text line: me S5e Dgreeable newd Sl yourtan, Mrs, N,\'m oelftre, dt-'
    '     Corrected text line: me the agreeable news of your and Mrs, H,\'s welfare, to-'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "me S5e Dgreeable newd" to "me the agreeable news", "Sl yourtan," to "of your and", "Mrs, N,\'m oelftre," to "Mrs, H,\'s welfare,".'

    '\n4. OCR Error from the User: gethAr wier theSpucWSdasir:d awcouetAof sour approving ehe'
    '   The JSON output is: '
    '     Original text line: gethAr wier theSpucWSdasir:d awcouetAof sour approving ehe'
    '     Corrected text line: gether with the much-desired account of your approving the'
    '     Confidence (%): 91 '
    '     Justification: Corrected the substitutions "gethAr wier" to "gether with", "theSpucWSdasir:d" to "the much-desired", "awcouetAof" to "account of", "sour approving" to "your approving", "ehe" to "the".'

    '\n5. OCR Error from the User: wc7emep inclosed iy fy lbst. Indee.t I did got doebxxofnhts'
    '   The JSON output is: '
    '     Original text line: wc7emep inclosed iy fy lbst. Indee.t I did got doebxxofnhts'
    '     Corrected text line: scheme, inclosed in my last. Indeed, I did not doubt of its'
    '     Confidence (%): 89 '
    '     Justification: Corrected the substitutions "wc7emep" to "scheme,", "inclosed iy" to "inclosed in", "fy lbst." to "my last.", "Indee.t" to "Indeed,", "I did got doebxxofnhts" to "I did not doubt of its".'

    '\n6. OCR Error from the User: meeting;hwith a ca:di  ec1eptiovt from bly 7hdse petsonf to'
    '   The JSON output is: '
    '     Original text line: meeting;hwith a ca:di  ec1eptiovt from bly 7hdse petsonf to'
    '     Corrected text line: meeting; with a candid reception, from all those persons to'
    '     Confidence (%): 97 '
    '     Justification: Corrected the substitutions "meeting;hwith" to "meeting; with", "a ca:di  ec1eptiovt" to "a candid reception,", "from bly 7hdse" to "from all those", "petsonf" to "persons".'

    '\n7. OCR Error from the User: whlT i, wasarecommnndodgSTWeir knort concere for relie'
    '   The JSON output is: '
    '     Original text line: whlT i, wasarecommnndodgSTWeir knort concere for relie'
    '     Corrected text line: whom it was recommended. Their known concern for reli-'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "whlT i," to "whom it", "wasarecommnndodg" to "was recommended.", "STWeir knort" to "Their known", "concere" to "concern", "relie" to "reli-".'

    '\n8. OCR Error from the User: gion. gkvocg mi sidficient assuiancT, taatAnythihg 1a: be un-'
    '   The JSON output is: '
    '     Original text line: gion. gkvocg mi sidficient assuiancT, taatAnythihg 1a: be un-'
    '     Corrected text line: gion, giving me sufficient assurance, that nothing can be un-'
    '     Confidence (%): 72 '
    '     Justification: Corrected the substitutions "gion." to "gion,", "gkvocg mi" to "giving me", "sidficient assuiancT," to "sufficient assurance,", "taatAnythihg 1a:" to "that nothing can be".'

    '\n9. OCR Error from the User: aVcept:blA sA thed, whech apy way tgnded to proaofu th Sr'
    '   The JSON output is: '
    '     Original text line: aVcept:blA sA thed, whech apy way tgnded to proaofu th Sr'
    '     Corrected text line: acceptable to them, which any way tended to promote their'
    '     Confidence (%): 61 '
    '     Justification: Corrected the substitutions "aVcept:blA sA" to "acceptable to", "thed," to "them,", "whech apy way tgnded" to "which any way tended", "proaofu th Sr" to "promote their".'

    '\n10. OCR Error from the User: improvemext Wn thD divito life.DIhamuIc b; lonfvssRp, indeed,'
    '   The JSON output is: '
    '     Original text line: improvemext Wn thD divito life.DIhamuIc b; lonfvssRp, indeed,'
    '     Corrected text line: improvement in the divine life. It must be confessed, indeed,'
    '     Confidence (%): 45 '
    '     Justification: Corrected the substitutions "improvemext Wn thD" to "improvement in the", "divito life.DIhamuIc b;" to "divine life. It must be", "lonfvssRp, indeed," to "confessed, indeed,".'
)



EXAMPLES_O_25 = (
    '\n1. OCR Error from the User: Dekr Sir,,Oxou, ST-t.v1e, 1734.'
    '   The JSON output is: '
    '     Original text line: Dekr Sir,,Oxou, ST-t.v1e, 1734.'
    '     Corrected text line: Dear Sir, Oxon, Sept. 17, 1734.'
    '     Confidence (%): 95 '
    '     Justification: Corrected the misrecognized characters "Dekr" to "Dear", "Sir,," to "Sir,", "Oxou," to "Oxon,", "ST-t.v1e," to "Sept. 17,".'

    '\n2. OCR Error from the User: IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought'
    '   The JSON output is: '
    '     Original text line: IaNid theDf:coua of your l.tterylast Fridih, whRchvDrought'
    '     Corrected text line: I Had the favour of your letter last Friday, which brought'
    '     Confidence (%): 94 '
    '     Justification: Corrected the substitutions "IaNid" to "I Had", "theDf:coua" to "the favour", "l.tterylast" to "letter last", "Fridih," to "Friday,", and "whRchvDrought" to "which brought". '

    '\n3. OCR Error from the User: me S5e Dgreeable newd Sl yourtan, Mrs, N,\'m oelftre, dt-'
    '   The JSON output is: '
    '     Original text line: me S5e Dgreeable newd Sl yourtan, Mrs, N,\'m oelftre, dt-'
    '     Corrected text line: me the agreeable news of your and Mrs, H,\'s welfare, to-'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "S5e" to "the", "Dgreeable newd" to "agreeable news", "Sl yourtan," to "of your and", "Mrs, N,\'m oelftre," to "Mrs, H,\'s welfare,".'

    '\n4. OCR Error from the User: gethAr wier theSpucWSdasir:d awcouetAof sour approving ehe'
    '   The JSON output is: '
    '     Original text line: gethAr wier theSpucWSdasir:d awcouetAof sour approving ehe'
    '     Corrected text line: gether with the much-desired account of your approving the'
    '     Confidence (%): 91 '
    '     Justification: Corrected the substitutions "gethAr" to "gether", "wier" to "with", "theSpucWSdasir:d" to "the much-desired", "awcouetAof" to "account of", "sour approving ehe" to "your approving the". '

    '\n5. OCR Error from the User: wc7emep inclosed iy fy lbst. Indee.t I did got doebxxofnhts'
    '   The JSON output is: '
    '     Original text line: wc7emep inclosed iy fy lbst. Indee.t I did got doebxxofnhts'
    '     Corrected text line: scheme, inclosed in my last. Indeed, I did not doubt of its'
    '     Confidence (%): 89 '
    '     Justification: Corrected the substitutions "wc7emep" to "scheme,", "inclosed iy fy lbst." to "inclosed in my last.", "Indee.t" to "Indeed,", "I did got doebxxofnhts" to "I did not doubt of its". '

    '\n6. OCR Error from the User: meeting;hwith a ca:di ec1eptiovt from bly 7hdse petsonf to'
    '   The JSON output is: '
    '     Original text line: meeting;hwith a ca:di ec1eptiovt from bly 7hdse petsonf to'
    '     Corrected text line: meeting; with a candid reception, from all those persons to'
    '     Confidence (%): 97 '
    '     Justification: Corrected the substitutions "meeting;hwith" to "meeting; with", "a ca:di ec1eptiovt" to "a candid reception,", "from bly 7hdse petsonf" to "from all those persons". '

    '\n7. OCR Error from the User: whlT i, wasarecommnndodgSTWeir knort concere for relie'
    '   The JSON output is: '
    '     Original text line: whlT i, wasarecommnndodgSTWeir knort concere for relie'
    '     Corrected text line: whom it was recommended. Their known concern for reli-'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "whlT i," to "whom it", "wasarecommnndodgSTWeir" to "was recommended. Their", "knort concere for relie" to "known concern for reli-". '

    '\n8. OCR Error from the User: gion. gkvocg mi sidficient assuiancT, taatAnythihg 1a: be un-'
    '   The JSON output is: '
    '     Original text line: gion. gkvocg mi sidficient assuiancT, taatAnythihg 1a: be un-'
    '     Corrected text line: gion, giving me sufficient assurance, that nothing can be un-'
    '     Confidence (%): 72 '
    '     Justification: Corrected the substitutions "gion. gkvocg mi" to "gion, giving me", "sidficient assuiancT," to "sufficient assurance,", "taatAnythihg 1a: be un-" to "that nothing can be un-". '

    '\n9. OCR Error from the User: aVcept:blA sA thed, whech apy way tgnded to proaofu th Sr'
    '   The JSON output is: '
    '     Original text line: aVcept:blA sA thed, whech apy way tgnded to proaofu th Sr'
    '     Corrected text line: acceptable to them, which any way tended to promote their'
    '     Confidence (%): 61 '
    '     Justification: Corrected the substitutions "aVcept:blA sA thed," to "acceptable to them,", "whech apy way tgnded to proaofu th Sr" to "which any way tended to promote their". '

    '\n10. OCR Error from the User: improvemext Wn thD divito life.DIhamuIc b; lonfvssRp, indeed,'
    '   The JSON output is: '
    '     Original text line: improvemext Wn thD divito life.DIhamuIc b; lonfvssRp, indeed,'
    '     Corrected text line: improvement in the divine life. It must be confessed, indeed,'
    '     Confidence (%): 45 '
    '     Justification: Corrected the substitutions "improvemext Wn thD" to "improvement in the", "divito life.DIhamuIc b;" to "divine life. It must be", "lonfvssRp, indeed," to "confessed, indeed,". '

    '\n11. OCR Error from the User: xs youuvery RstlyTobserved, th,tRwe mustnhbke a graatiiCo-'
    '   The JSON output is: '
    '     Original text line: xs youuvery RstlyTobserved, th,tRwe mustnhbke a graatiiCo-'
    '     Corrected text line: as you very justly observed, that we must make a great pro-'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "xs youuvery RstlyTobserved," to "as you very justly observed,", "th,tRwe mustnhbke a graatiiCo-" to "that we must make a great pro-". '

    '\n12. OCR Error from the User: gress imy,sovgiot, and le inured bb fJzquRnt prayerOanc medi-'
    '   The JSON output is: '
    '     Original text line: gress imy,sovgiot, and le inured bb fJzquRnt prayerOanc medi-'
    '     Corrected text line: gress in religion, and be inured by frequent prayer and medi-'
    '     Confidence (%): 81 '
    '     Justification: Corrected the substitutions "gress imy,sovgiot," to "gress in religion,", "and le inured bb fJzquRnt" to "and be inured by frequent", "prayerOanc medi-" to "prayer and medi-". '

    '\n13. OCR Error from the User: tation,hfG the exuatic cNn.empla,aCn of heaveniygobja:es, beSyre'
    '   The JSON output is: '
    '     Original text line: tation,hfG the exuatic cNn.empla,aCn of heaveniygobja:es, beSyre'
    '     Corrected text line: tation, to the extatic contemplation of heavenly objects, before'
    '     Confidence (%): 90 '
    '     Justification: Corrected the substitutions "tation,hfG" to "tation, to", "the exuatic cNn.empla,aCn" to "the extatic contemplation", "of heaveniygobja:es, beSyre" to "of heavenly objects, before". '

    '\n14. OCR Error from the User: we5uan arhivexat trvg hxaveOlrSsindodnbss; ann, perh,ps, after'
    '   The JSON output is: '
    '     Original text line: we5uan arhivexat trvg hxaveOlrSsindodnbss; ann, perh,ps, after'
    '     Corrected text line: we can arrive at true heavenly-mindedness; and, perhaps, after'
    '     Confidence (%): 75 '
    '     Justification: Corrected the substitutions "we5uan arhivexat" to "we can arrive at", "trvg hxaveOlrSsindodnbss;" to "true heavenly-mindedness;", "ann, perh,ps, after" to "and, perhaps, after". '

    '\n15. OCR Error from the User: amlhohC endbavours, toilstOouo souls yre ivmegsed it thesk'
    '   The JSON output is: '
    '     Original text line: amlhohC endbavours, toilstOouo souls yre ivmegsed it thesk'
    '     Corrected text line: all our endeavours, whilst our souls are immersed in these'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "amlhohC endbavours," to "all our endeavours,", "toilstOouo souls" to "whilst our souls", "yre ivmegsed it thesk" to "are immersed in these". '

    '\n16. OCR Error from the User: fleshly taberRaulvs, wh swall5plke but vgdy smallpafcanceopin uo'
    '   The JSON output is: '
    '     Original text line: fleshly taberRaulvs, wh swall5plke but vgdy smallpafcanceopin uo'
    '     Corrected text line: fleshly tabernacles, we shall make but very small advances in so'
    '     Confidence (%): 74 '
    '     Justification: Corrected the substitutions "fleshly taberRaulvs," to "fleshly tabernacles,", "wh swall5plke" to "we shall make", "but vgdy smallpafcanceopin uo" to "but very small advances in so". '

    '\n17. OCR Error from the User: .elighfful and glSriour aO undSutae.n .ABut bflievx me,hSi.,'
    '   The JSON output is: '
    '     Original text line: .elighfful and glSriour aO undSutae.n .ABut bflievx me,hSi.,'
    '     Corrected text line: delightful and glorious an undertaking. But believe me, Sir,'
    '     Confidence (%): 70 '
    '     Justification: Corrected the substitutions ".elighfful and glSriour" to "delightful and glorious", "aO undSutae.n .ABut" to "an undertaking. But", "bflievx me,hSi.," to "believe me, Sir,". '

    '\n18. OCR Error from the User: you cannod imegine,Sprw vaCtly1oewviceable uhe  onItant u., oa'
    '   The JSON output is: '
    '     Original text line: you cannod imegine,Sprw vaCtly1oewviceable uhe  onItant u., oa'
    '     Corrected text line: you cannot imagine, how vastly serviceable the constant use of'
    '     Confidence (%): 84 '
    '     Justification: Corrected the substitutions "you cannod imegine," to "you cannot imagine,", "Sprw vaCtly1oewviceable uhe" to "how vastly serviceable the", "onItant u., oa" to "constant use of". '

    '\n19. OCR Error from the User: all the means;.i weligion willabe. rn Ccquiring thii fueC,;f habit'
    '   The JSON output is: '
    '     Original text line: all the means;.i weligion willabe. rn Ccquiring thii fueC,;f habit'
    '     Corrected text line: all the means of religion will be, in acquiring this blessed habit'
    '     Confidence (%): 71 '
    '     Justification: Corrected the substitutions "all the means;.i" to "all the means of", "weligion willabe." to "religion will be,", "rn Ccquiring thii fueC,;f habit" to "in acquiring this blessed habit". '

    '\n20. OCR Error from the User: od mind.tSuch,OptOan early 1isingvio tiA morning, pwuwic'
    '   The JSON output is: '
    '     Original text line: od mind.tSuch,OptOan early 1isingvio tiA morning, pwuwic'
    '     Corrected text line: of mind. Such, as an early rising in the morning, public'
    '     Confidence (%): 88 '
    '     Justification: Corrected the substitutions "od mind.tSuch,OptOan" to "of mind. Such, as an", "early 1isingvio tiA morning," to "early rising in the morning,", "pwuwic" to "public". '

    '\n21. OCR Error from the User: anddprivaxe prayer,fa duh tifptrahce in all;thi gh,waW fret'
    '   The JSON output is: '
    '     Original text line: anddprivaxe prayer,fa duh tifptrahce in all;thi gh,waW fret'
    '     Corrected text line: and private prayer, a due temperance in all things, and fre-'
    '     Confidence (%): 75 '
    '     Justification: Corrected the substitutions "anddprivaxe" to "and private", "prayer,fa duh tifptrahce" to "prayer, a due temperance", "in all;thi gh,waW fret" to "in all things, and fre-". '

    '\n22. OCR Error from the User: quent meditatigwUontlhe ,nfinite kovJ tpd puritb of that aId'
    '   The JSON output is: '
    '     Original text line: quent meditatigwUontlhe ,nfinite kovJ tpd puritb of that aId'
    '     Corrected text line: quent meditation on the infinite love and purity of that un-'
    '     Confidence (%): 69 '
    '     Justification: Corrected the substitutions "quent meditatigwUontlhe" to "quent meditation on the", ",nfinite kovJ tpd puritb" to "infinite love and purity", "of that aId" to "of that un-". '

    '\n23. OCR Error from the User: pa.alleAed ,atteDs oS all prrbectionp hu1 dea1 Redtemer. At'
    '   The JSON output is: '
    '     Original text line: pa.alleAed ,atteDs oS all prrbectionp hu1 dea1 Redtemer. At'
    '     Corrected text line: paralleled pattern of all perfection, our dear Redeemer. As'
    '     Confidence (%): 78 '
    '     Justification: Corrected the substitutions "pa.alleAed" to "paralleled", ",atteDs oS all prrbectionp" to "pattern of all perfection,", "hu1 dea1 Redtemer. At" to "our dear Redeemer. As". '

    '\n24. OCR Error from the User: boh your 1enyidtini, Svfr the deg,neracy of Whe age, aW thd'
    '   The JSON output is: '
    '     Original text line: boh your 1enyidtini, Svfr the deg,neracy of Whe age, aW thd'
    '     Corrected text line: for your mentioning, Sir, the degeneracy of the age, as the'
    '     Confidence (%): 82 '
    '     Justification: Corrected the substitutions "boh your 1enyidtini," to "for your mentioning,", "Svfr the deg,neracy" to "Sir, the degeneracy", "of Whe age, aW thd" to "of the age, as the". '

    '\n25. OCR Error from the User: lras dbjection agaicstgour ma ing,fPrwhe cadvayces in ynv'
    '   The JSON output is: '
    '     Original text line: lras dbjection agaicstgour ma ing,fPrwhe cadvayces in ynv'
    '     Corrected text line: least objection against our making further advances in any'
    '     Confidence (%): 73 '
    '     Justification: Corrected the substitutions "lras dbjection" to "least objection", "agaicstgour ma ing,fPrwhe" to "against our making further", "cadvayces in ynv" to "advances in any". '
)


EXAMPLES_F_25 = (
    '\n1. OCR Error from the User: onlh fgrgthe putliik usen tnhesr by papticb-'
    '   The JSON output is: '
    '     Original text line: onlh fgrgthe putliik usen tnhesr by papticb-'
    '     Corrected text line: only for the publick use, unless by particu-'
    '     Confidence (%): 95 '
    '     Justification: Corrected substitutions "onlh" to "only", "fgrgthe" to "for the", "putliik" to "publick", '
    '"tnhesr" to "unless", and "papticb-" to "particu-". '

    '\n2. OCR Error from the User: )owR a Bvrrelvof Flintd yith SheVArms, .J'
    '   The JSON output is: '
    '     Original text line: )owR a Bvrrelvof Flintd yith SheVArms, .J'
    '     Corrected text line: down a Barrel of Flints with the Arms, to'
    '     Confidence (%): 94 '
    '     Justification: Corrected substitutions "owR" to "down", "Bvrrel" to "Barrel", "Flintd" to '
    '"Flints", "yith" to "with", "SheVArms," to "the Arms," and removed the incorrect characters at the end. '

    '\n3. OCR Error from the User: Poaneation of nharlesIS.llarsV- tht restktFbCaCtais'
    '   The JSON output is: '
    '     Original text line: Poaneation of nharlesIS.llarsV- tht restktFbCaCtais'
    '     Corrected text line: Plantation of Charles Sellars - the rest to Captain'
    '     Confidence (%): 83 '
    '     Justification: Corrected substitutions "Poaneation" to "Plantation", "nharlesIS.llarsV-" to '
    '"Charles Sellars -", "tht" to "the", "restktFbCaCtais" to "rest to Captain". '

    '\n4. OCR Error from the User: The OfgiceDs w7h cage dswn'
    '   The JSON output is: '
    '     Original text line: The OfgiceDs w7h cage dswn'
    '     Corrected text line: The Officers who came down'
    '     Confidence (%): 91 '
    '     Justification: Corrected substitutions "OfgiceDs" to "Officers", "w7h" to "who", "cage" to '
    '"came", "dswn" to "down". '

    '\n5. OCR Error from the User: to Rkpair to Captiin Hop sdBmppany win5Ueight'
    '   The JSON output is: '
    '     Original text line: to Rkpair to Captiin Hop sdBmppany win5Ueight'
    '     Corrected text line: to Repair to Captain Hoggs Company with eight'
    '     Confidence (%): 89 '
    '     Justification: Corrected substitutions "Rkpair" to "Repair", "Captiin" to "Captain", '
    '"Hop sdBmppany" to "Hoggs Company", "win5Ueight" to "with eight". '

    '\n6. OCR Error from the User: onea,;rsean,i one Co,poJae, onD Drummer,'
    '   The JSON output is: '
    '     Original text line: onea,;rsean,i one Co,poJae, onD Drummer,'
    '     Corrected text line: one Sergeant, one Corporal, one Drummer,'
    '     Confidence (%): 97 '
    '     Justification: Corrected substitutions "onea,;rsean,i" to "one Sergeant,", "one Co,poJae," '
    'to "one Corporal,", "onD" to "one", and restored "Drummer,". '

    '\n7. OCR Error from the User: widhouV aWSufgeon, ifbQou .ill co thft luny, an'
    '   The JSON output is: '
    '     Original text line: widhouV aWSufgeon, ifbQou .ill co thft luny, an'
    '     Corrected text line: without a Surgeon, if you will do that duty, an'
    '     Confidence (%): 85 '
    '     Justification: Corrected substitutions "widhouV" to "without", "aWSufgeon," to "a Surgeon,", '
    '"ifbQou" to "if you", ".ill co" to "will do", "thft luny," to "that duty,". '

    '\n8. OCR Error from the User: hclt trereRuntic he soins, inedrpeu to escTyV the'
    '   The JSON output is: '
    '     Original text line: hclt trereRuntic he soins, inedrpeu to escTyV the'
    '     Corrected text line: halt there until he joins, in order to escort the'
    '     Confidence (%): 72 '
    '     Justification: Corrected substitutions "hclt" to "halt", "trereRuntic" to "there until", '
    '"he soins," to "he joins,", "inedrpeu" to "in order", "escTyV" to "escort". '

    '\n9. OCR Error from the User: arR totsupplw hif vrom th-'
    '   The JSON output is: '
    '     Original text line: arR totsupplw hif vrom th-'
    '     Corrected text line: are to supply him from the'
    '     Confidence (%): 61 '
    '     Justification: Corrected substitutions "arR" to "are", "totsupplw" to "to supply", '
    '"hif" to "him", "vrom" to "from". '

    '\n10. OCR Error from the User: marce chym immLdgateyu to Fhrt Aumbe;land;'
    '   The JSON output is: '
    '     Original text line: marce chym immLdgateyu to Fhrt Aumbe;land;'
    '     Corrected text line: march them immediately to Fort Cumberland;'
    '     Confidence (%): 45 '
    '     Justification: Corrected substitutions "marce" to "march", "chym" to "them", "immLdgateyu" '
    'to "immediately", "Fhrt Aumbe;land;" to "Fort Cumberland;". '

    '\n11. OCR Error from the User: Wincce,rerv andAafoul two thousenQIweight'
    '   The JSON output is: '
    '     Original text line: Wincce,rerv andAafoul two thousenQIweight'
    '     Corrected text line: Winchester, and about two thousand weight'
    '     Confidence (%): 90 '
    '     Justification: Corrected substitutions "Wincce,rerv" to "Winchester,", "andAafoul" to '
    '"and about", "thousenQIweight" to "thousand weight". '

    '\n12. OCR Error from the User: Captkin Asiby and CcmpDnyC xtSdhe'
    '   The JSON output is: '
    '     Original text line: Captkin Asiby and CcmpDnyC xtSdhe'
    '     Corrected text line: Captain Ashby and Company, at the'
    '     Confidence (%): 95 '
    '     Justification: Corrected substitutions "Captkin" to "Captain", "Asiby" to "Ashby", '
    '"CcmpDnyC" to "Company,", "xtSdhe" to "at the". '

    '\n13. OCR Error from the User: 28th Winchesteh::Wc7obec 28tW,7$7.5.'
    '   The JSON output is: '
    '     Original text line: 28th Winchesteh::Wc7obec 28tW,7$7.5.'
    '     Corrected text line: 28th Winchester: October 28th, 1755.'
    '     Confidence (%): 92 '
    '     Justification: Corrected substitutions "Winchesteh::Wc7obec" to "Winchester: October", '
    '"28tW,7$7.5." to "28th, 1755.". '

    '\n14. OCR Error from the User: The Offirerl wuo ,aoe dowu'
    '   The JSON output is: '
    '     Original text line: The Offirerl wuo ,aoe dowu'
    '     Corrected text line: The Officers who came down'
    '     Confidence (%): 89 '
    '     Justification: Corrected substitutions "Offirerl" to "Officers", "wuo" to "who", ",aoe" to '
    '"came", "dowu" to "down". '

    '\n15. OCR Error from the User: 28thelTo EgSign Flemixg, oO tJe Viegiaia RrtiTent.'
    '   The JSON output is: '
    '     Original text line: 28thelTo EgSign Flemixg, oO tJe Viegiaia RrtiTent.'
    '     Corrected text line: 28th. To Ensign Fleming, of the Virginia Regiment.'
    '     Confidence (%): 85 '
    '     Justification: Corrected substitutions "28thelTo" to "28th. To", "EgSign" to "Ensign", '
    '"Flemixg," to "Fleming,", "oO tJe Viegiaia RrtiTent." to "of the Virginia Regiment.". '

    '\n16. OCR Error from the User: CaptainwH go\'s  sipanyIbt FortBDinwiDdie'
    '   The JSON output is: '
    '     Original text line: CaptainwH go\'s  sipanyIbt FortBDinwiDdie'
    '     Corrected text line: Captain Hogg\'s Company at Fort Dinwiddie'
    '     Confidence (%): 81 '
    '     Justification: Corrected substitutions "CaptainwH go\'s" to "Captain Hogg\'s", "sipanyIbt" '
    'to "Company at", "FortBDinwiDdie" to "Fort Dinwiddie". '

    '\n17. OCR Error from the User: I receiv,l yIuds5of the 6thr gfnOctc-'
    '   The JSON output is: '
    '     Original text line: I receiv,l yIuds5of the 6thr gfnOctc-'
    '     Corrected text line: I received yours of the 6th. of Octo-'
    '     Confidence (%): 88 '
    '     Justification: Corrected substitutions "receiv,l" to "received", "yIuds5of" to "yours of", '
    '"6thr" to "6th.", "gfnOctc-" to "of Octo-". '

    '\n18. OCR Error from the User: W Ichester, and a,out tpo trCusgndhoeihht'
    '   The JSON output is: '
    '     Original text line: W Ichester, and a,out tpo trCusgndhoeihht'
    '     Corrected text line: Winchester, and about two thousand weight'
    '     Confidence (%): 84 '
    '     Justification: Corrected substitutions "W Ichester," to "Winchester,", "a,out tpo" to '
    '"and about two", "trCusgndhoeihht" to "thousand weight". '

    '\n19. OCR Error from the User: RDcruttiOo; and shey arec:llowed un.il tle 1ux. ohhDe-'
    '   The JSON output is: '
    '     Original text line: RDcruttiOo; and shey arec:llowed un.il tle 1ux. ohhDe-'
    '     Corrected text line: Recruiting; and they are allowed until the 1st. of De-'
    '     Confidence (%): 78 '
    '     Justification: Corrected substitutions "RDcruttiOo;" to "Recruiting;", "shey" to "they", '
    '"arec:llowed" to "are allowed", "un.il tle 1ux. ohhDe-" to "until the 1st. of De-". '

    '\n20. OCR Error from the User: wach 1aR pistiWguishes'
    '   The JSON output is: '
    '     Original text line: wach 1aR pistiWguishes'
    '     Corrected text line: each man distinguishes'
    '     Confidence (%): 91 '
    '     Justification: Corrected substitutions "wach 1aR" to "each man", "pistiWguishes" to "distinguishes". '

    '\n21. OCR Error from the User: cbqtaln Ashbl and C-mpany,nat tpe'
    '   The JSON output is: '
    '     Original text line: cbqtaln Ashbl and C-mpany,nat tpe'
    '     Corrected text line: Captain Ashby and Company, at the'
    '     Confidence (%): 93 '
    '     Justification: Corrected substitutions "cbqtaln" to "Captain", "Ashbl" to "Ashby", '
    '"C-mpany,nat tpe" to "Company, at the". '

    '\n22. OCR Error from the User: TlontatiFn of  rarles SellaDs - thJ eest:uoVCaprain'
    '   The JSON output is: '
    '     Original text line: TlontatiFn of  rarles SellaDs - thJ eest:uoVCaprain'
    '     Corrected text line: Plantation of Charles Sellars - the rest to Captain'
    '     Confidence (%): 87 '
    '     Justification: Corrected substitutions "TlontatiFn" to "Plantation", "rarles SellaDs" to '
    '"Charles Sellars", "thJ eest:uoVCaprain" to "the rest to Captain". '

    '\n23. OCR Error from the User: WTeACommissary As touspe'
    '   The JSON output is: '
    '     Original text line: WTeACommissary As touspe'
    '     Corrected text line: The Commissary is to see'
    '     Confidence (%): 92 '
    '     Justification: Corrected substitutions "WTeACommissary As touspe" to "The Commissary is to see". '

    '\n24. OCR Error from the User: If a,y have rwcviced SeOgAantsBpay,Uii'
    '   The JSON output is: '
    '     Original text line: If a,y have rwcviced SeOgAantsBpay,Uii'
    '     Corrected text line: If any have received Sergeants pay, it'
    '     Confidence (%): 90 '
    '     Justification: Corrected substitutions "If a,y" to "If any", "rwcviced" to "received", '
    '"SeOgAantsBpay,Uii" to "Sergeants pay, it". '

    '\n25. OCR Error from the User: WhenrwAd:xrrive aw Winchestlr, yom eost p1ocidedyour men witf Card'
    '   The JSON output is: '
    '     Original text line: WhenrwAd:xrrive aw Winchestlr, yom eost p1ocidedyour men witf Card'
    '     Corrected text line: When you arrive at Winchester, you must provide your men with Car-'
    '     Confidence (%): 86 '
    '     Justification: Corrected substitutions "WhenrwAd:xrrive aw" to "When you arrive at", '
    '"Winchestlr," to "Winchester,", "yom eost p1ocidedyour" to "you must provide your", "witf" to "with". '
)

EXAMPLES_F_50 = (
    '\n1. OCR Error from the User: only foh the cublick uS;, uJle,s fa paetsdu-'
    '   The JSON output is: '
    '     Original text line: only foh the cublick uS;, uJle,s fa paetsdu-'
    '     Corrected text line: only for the publick use, unless by particu-'
    '     Confidence (%): 95 '
    '     Justification: Corrected the misrecognized characters "foh" to "for", "cublick uS;" to "publick use,", '
    '"uJle,s" to "unless", and "fa paetsdu-" to "by particu-". '

    '\n2. OCR Error from the User: dowO - BaruelMofgFlinWs wrth ihA5Arms, to'
    '   The JSON output is: '
    '     Original text line: dowO - BaruelMofgFlinWs wrth ihA5Arms, to'
    '     Corrected text line: down a Barrel of Flints with the Arms, to'
    '     Confidence (%): 94 '
    '     Justification: Corrected the substitutions "dowO" to "down", "BaruelMofgFlinWs" to "Barrel of Flints", '
    '"wrth" to "with", and "ihA5Arms" to "the Arms, to". '

    '\n3. OCR Error from the User: ulanaation of Rharleigeellurs - the restoio BaptDir'
    '   The JSON output is: '
    '     Original text line: ulanaation of Rharleigeellurs - the restoio BaptDir'
    '     Corrected text line: Plantation of Charles Sellars - the rest to Captain'
    '     Confidence (%): 83 '
    '     Justification: Corrected the misrecognized characters "ulanaation" to "Plantation", "Rharleigeellurs" to '
    '"Charles Sellars", "restoio" to "rest to", and "BaptDir" to "Captain". '

    '\n4. OCR Error from the User: Tce OffScers:Nho .ame1down'
    '   The JSON output is: '
    '     Original text line: Tce OffScers:Nho .ame1down'
    '     Corrected text line: The Officers who came down'
    '     Confidence (%): 91 '
    '     Justification: Corrected the substitutions "Tce" to "The", "OffScers:Nho" to "Officers who", and '
    '".ame1down" to "came down". '

    '\n5. OCR Error from the User: toRRepvrr to CaptainUHorgseComtanyiwitd  cght'
    '   The JSON output is: '
    '     Original text line: toRRepvrr to CaptainUHorgseComtanyiwitd  cght'
    '     Corrected text line: to Repair to Captain Hoggs Company with eight'
    '     Confidence (%): 89 '
    '     Justification: Corrected the substitutions "toRRepvrr" to "to Repair", "CaptainUHorgseComtanyiwitd" to '
    '"Captain Hoggs Company with", and "cght" to "eight". '

    '\n6. OCR Error from the User: one Se1Wignt,duIe Corporal,Vhne Drummee,'
    '   The JSON output is: '
    '     Original text line: one Se1Wignt,duIe Corporal,Vhne Drummee,'
    '     Corrected text line: one Sergeant, one Corporal, one Drummer,'
    '     Confidence (%): 97 '
    '     Justification: Corrected the misrecognized characters "Se1Wignt,duIe" to "Sergeant, one", '
    '"Vhne" to "one", and "Drummee" to "Drummer". '

    '\n7. OCR Error from the User: eit5ous a Curgeon, in yDu wilH doedbat dutr, an'
    '   The JSON output is: '
    '     Original text line: eit5ous a Curgeon, in yDu wilH doedbat dutr, an'
    '     Corrected text line: without a Surgeon, if you will do that duty, an'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "eit5ous" to "without", "Curgeon" to "Surgeon", '
    '"in yDu wilH" to "if you will", "doedbat dutr" to "do that duty", and "an" to "an". '

    '\n8. OCR Error from the User: halm therJ fntil be soins, bn.\'pde; to etcsrt the'
    '   The JSON output is: '
    '     Original text line: halm therJ fntil be soins, bn.\'pde; to etcsrt the'
    '     Corrected text line: halt there until he joins, in order to escort the'
    '     Confidence (%): 72 '
    '     Justification: Corrected the misrecognized characters "halm" to "halt", "therJ fntil be soins" to '
    '"there until he joins", "bn.\'pde;" to "in order", and "etcsrt" to "escort". '

    '\n9. OCR Error from the User: xrJ to supply himbfcoi xhe'
    '   The JSON output is: '
    '     Original text line: xrJ to supply himbfcoi xhe'
    '     Corrected text line: are to supply him from the'
    '     Confidence (%): 61 '
    '     Justification: Corrected the substitutions "xrJ" to "are", "himbfcoi" to "him from", and "xhe" to "the". '

    '\n10. OCR Error from the User: marcoethemaimnediamelyxto  or7RCumberland'
    '   The JSON output is: '
    '     Original text line: marcoethemaimnediamelyxto  or7RCumberland'
    '     Corrected text line: march them immediately to Fort Cumberland;'
    '     Confidence (%): 45 '
    '     Justification: Corrected the misrecognized characters "marcoethemaimnediamelyxto" to "march them '
    'immediately to" and "or7RCumberland" to "Fort Cumberland". '

    '\n11. OCR Error from the User: Wincheshex, and abour xwI VhouiandMweiohS'
    '   The JSON output is: '
    '     Original text line: Wincheshex, and abour xwI VhouiandMweiohS'
    '     Corrected text line: Winchester, and about two thousand weight'
    '     Confidence (%): 90 '
    '     Justification: Corrected the misrecognized characters "Wincheshex" to "Winchester", "abour" to "about", '
    '"xwI VhouiandMweiohS" to "two thousand weight". '

    '\n12. OCR Error from the User: Captain Asmby aTb Oompani, dtkthb'
    '   The JSON output is: '
    '     Original text line: Captain Asmby aTb Oompani, dtkthb'
    '     Corrected text line: Captain Ashby and Company, at the'
    '     Confidence (%): 95 '
    '     Justification: Corrected the substitutions "Asmby" to "Ashby", "aTb" to "and", "Oompani" to '
    '"Company", and "dtkthb" to "at the". '

    '\n13. OCR Error from the User: 28ap Winehkster:  ctoberfm8th, 17c5T'
    '   The JSON output is: '
    '     Original text line: 28ap Winehkster:  ctoberfm8th, 17c5T'
    '     Corrected text line: 28th Winchester: October 28th, 1755.'
    '     Confidence (%): 92 '
    '     Justification: Corrected the substitutions "28ap" to "28th", "Winehkster" to "Winchester", '
    '"ctoberfm8th" to "October 28th", and "17c5T" to "1755". '

    '\n14. OCR Error from the User: Padole Hampt J.'
    '   The JSON output is: '
    '     Original text line: Padole Hampt J.'
    '     Corrected text line: Parole Hampton.'
    '     Confidence (%): 89 '
    '     Justification: Corrected the substitutions "Padole" to "Parole", and "Hampt J." to "Hampton.". '

    '\n15. OCR Error from the User:  roy Foru tumbgrlvnd with Cooonec'
    '   The JSON output is: '
    '     Original text line:  roy Foru tumbgrlvnd with Cooonec'
    '     Corrected text line: from Fort Cumberland with Colonel'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "roy" to "from", "Foru tumbgrlvnd" to "Fort Cumberland", '
    'and "Cooonec" to "Colonel". '

    '\n16. OCR Error from the User: Washihgton,1are igme;i:Veny to go RRlrwi-'
    '   The JSON output is: '
    '     Original text line: Washihgton,1are igme;i:Veny to go RRlrwi-'
    '     Corrected text line: Washington, are immediately to go Recrui-'
    '     Confidence (%): 88 '
    '     Justification: Corrected the substitutions "Washihgton" to "Washington", "igme;i:Veny" to "immediately", '
    'and "RRlrwi-" to "Recrui-". '

    '\n17. OCR Error from the User: tane; mnd Whe, aye atlowed yntil Ie, 1st. ofpDe-'
    '   The JSON output is: '
    '     Original text line: tane; mnd Whe, aye atlowed yntil Ie, 1st. ofpDe-'
    '     Corrected text line: ting; and they are allowed until the 1st. of De-'
    '     Confidence (%): 82 '
    '     Justification: Corrected the substitutions "tane;" to "ting;", "mnd Whe" to "and they", "aye" to "are", '
    '"atlowed" to "allowed", "yntil Ie" to "until the", and "ofpDe-" to "of De-". '

    '\n18. OCR Error from the User: cembers at wnich teme wfhShel dornof'
    '   The JSON output is: '
    '     Original text line: cembers at wnich teme wfhShel dornof'
    '     Corrected text line: cember; at which time if they do not'
    '     Confidence (%): 84 '
    '     Justification: Corrected the substitutions "cembers" to "cember;", "at wnich" to "at which", "teme" to '
    '"time", "wfhShel" to "if they", and "dornof" to "do not". '

    '\n19. OCR Error from the User: pvnctuaoly aprearsat uhegpHace ot deldez-'
    '   The JSON output is: '
    '     Original text line: pvnctuaoly aprearsat uhegpHace ot deldez-'
    '     Corrected text line: punctually appear at the place of Rendez-'
    '     Confidence (%): 80 '
    '     Justification: Corrected the substitutions "pvnctuaoly" to "punctually", "aprearsat" to "appear at", '
    '"uhegpHace" to "the place", and "ot deldez-" to "of Rendez-". '

    '\n20. OCR Error from the User: vousVasshgJed tieo,othey wfol be tried5byra'
    '   The JSON output is: '
    '     Original text line: vousVasshgJed tieo,othey wfol be tried5byra'
    '     Corrected text line: vous assigned them, they will be tried by a'
    '     Confidence (%): 77 '
    '     Justification: Corrected the substitutions "vousVasshgJed" to "vous assigned", "tieo,othey" to "them, they", '
    '"wfol" to "will", and "tried5byra" to "tried by a". '

    '\n21. OCR Error from the User: Oourb MDstihl, for disobmdiencecof OdderW5'
    '   The JSON output is: '
    '     Original text line: Oourb MDstihl, for disobmdiencecof OdderW5'
    '     Corrected text line: Court Martial, for disobedience of Orders.'
    '     Confidence (%): 79 '
    '     Justification: Corrected the substitutions "Oourb" to "Court", "MDstihl" to "Martial", "disobmdiencecof" to '
    '"disobedience of", and "OdderW5" to "Orders.". '

    '\n22. OCR Error from the User: They arh to masg uqom phe Aid ne comp'
    '   The JSON output is: '
    '     Original text line: They arh to masg uqom phe Aid ne comp'
    '     Corrected text line: They are to wait upon the Aid de camp'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "arh" to "are", "masg" to "wait", "uqom" to "upon", "phe" to '
    '"the", "ne" to "de", and "comp" to "camp". '

    '\n23. OCR Error from the User: at oney f  he Clock, mo ricmbvn t Rir Recxui-'
    '   The JSON output is: '
    '     Original text line: at oney f  he Clock, mo ricmbvn t Rir Recxui-'
    '     Corrected text line: at one of the Clock, to receive their Recrui-'
    '     Confidence (%): 82 '
    '     Justification: Corrected the substitutions "oney f" to "one of", "he" to "the", "mo" to "to", "ricmbvn" to '
    '"receive", and "Rir Recxui-" to "their Recrui-". '

    '\n24. OCR Error from the User: ting Ivstructiotp, EachwOffu,er tresent, top,i s'
    '   The JSON output is: '
    '     Original text line: ting Ivstructiotp, EachwOffu,er tresent, top,i s'
    '     Corrected text line: ting Instructions. Each Officer present, to give'
    '     Confidence (%): 80 '
    '     Justification: Corrected the substitutions "Ivstructiotp" to "Instructions.", "EachwOffu,er" to '
    '"Each Officer", "tresent" to "present", and "top,i s" to "to give". '

    '\n25. OCR Error from the User: id x Retyrn imuefiately ov the numosl'
    '   The JSON output is: '
    '     Original text line: id x Retyrn imuefiately ov the numosl'
    '     Corrected text line: in a Return immediately of the number'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "id x" to "in a", "Retyrn" to "Return", "imuefiately" to'
    ' "immediately", "ov" to "of", and "numosl" to "number". '

    '\n26. OCR Error from the User: of men hp IaS Rnlintxp. - Onz Subartexn,'
    '   The JSON output is: '
    '     Original text line: of men hp IaS Rnlintxp. - Onz Subartexn,'
    '     Corrected text line: of men he has enlisted. - One Subaltern,'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "hp" to "he", "IaS" to "has", "Rnlintxp" to "enlisted.", '
    'and "Onz Subartexn" to "One Subaltern,". '

    '\n27. OCR Error from the User: obe SergLanc, Ane Corqo,al, ond DruTe;r,'
    '   The JSON output is: '
    '     Original text line: obe SergLanc, Ane Corqo,al, ond DruTe;r,'
    '     Corrected text line: one Sergeant, one Corporal, one Drummer,'
    '     Confidence (%): 88 '
    '     Justification: Corrected the substitutions "obe" to "one", "SergLanc" to "Sergeant,", "Ane" to "one", '
    '"Corqo,al" to "Corporal,", and "ond DruTe;r" to "one Drummer,". '

    '\n28. OCR Error from the User: bnd tlenIy5five provdtS men, aJe toUmounc'
    '   The JSON output is: '
    '     Original text line: bnd tlenIy5five provdtS men, aJe toUmounc'
    '     Corrected text line: and twenty five private men, are to mount'
    '     Confidence (%): 80 '
    '     Justification: Corrected the substitutions "bnd" to "and", "tlenIy5five" to "twenty five", "provdtS" to '
    '"private", "aJe" to "are", and "toUmounc" to "to mount". '

    '\n29. OCR Error from the User: Gperd tS - eas,Tana to bt relRevrd to - worrow'
    '   The JSON output is: '
    '     Original text line: Gperd tS - eas,Tana to bt relRevrd to - worrow'
    '     Corrected text line: Guard to - day, and to be relieved to - morrow'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "Gperd" to "Guard", "tS - eas," to "to - day,", "Tana" to "and", '
    '"bt" to "be", "relRevrd" to "relieved", and "worrow" to "mororrow". '

    '\n30. OCR Error from the User: at IedWo\'clock.x-.vAllhR poOtx add Returns'
    '   The JSON output is: '
    '     Original text line: at IedWo\'clock.x-.vAllhR poOtx add Returns'
    '     Corrected text line: at ten o\'clock. -. All Reports and Returns'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "IedWo\'clock.x-.v" to "ten o\'clock. -. ", "AllhR" to "All", '
    '"poOtx" to "Reports", and "add" to "and". '

    '\n31. OCR Error from the User: agk toDbe made uf thexAid de C,np.'
    '   The JSON output is: '
    '     Original text line: agk toDbe made uf thexAid de C,np.'
    '     Corrected text line: are to be made to the Aid de Camp.'
    '     Confidence (%): 82 '
    '     Justification: Corrected the substitutions "agk" to "are", "toDbe" to "to be", "made uf" to "made to", '
    '"thex" to "the", "C,np." to "Camp.". '

    '\n32. OCR Error from the User: 28Vh. Tm SsIign Fmevint, of the CirginaamRegirent.'
    '   The JSON output is: '
    '     Original text line: 28Vh. Tm SsIign Fmevint, of the CirginaamRegirent.'
    '     Corrected text line: 28th. To Ensign Fleming, of the Virginia Regiment.'
    '     Confidence (%): 88 '
    '     Justification: Corrected the substitutions "28Vh." to "28th.", "Tm SsIign" to "To Ensign", "Fmevint" to '
    '"Fleming,", and "CirginaamRegirent." to "Virginia Regiment.". '

    '\n33. OCR Error from the User: You ire herewy Frnbred do reyairWto'
    '   The JSON output is: '
    '     Original text line: You ire herewy Frnbred do reyairWto'
    '     Corrected text line: You are hereby ordered to repair to'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "ire" to "are", "herewy" to "hereby", "Frnbred" to "ordered", '
    'and "reyairWto" to "repair to". '

    '\n34. OCR Error from the User: C,ptain Ho g\'. CSmpaga Dt Fo-y Dnnwiddie'
    '   The JSON output is: '
    '     Original text line: C,ptain Ho g\'. CSmpaga Dt Fo-y Dnnwiddie'
    '     Corrected text line: Captain Hogg\'s Company at Fort Dinwiddie'
    '     Confidence (%): 81 '
    '     Justification: Corrected the substitutions "C,ptain" to "Captain", "Ho g\'." to "Hogg\'s", "CSmpaga" to '
    '"Company", "Dt Fo-y" to "at Fort", and "Dnnwiddie" to "Dinwiddie". '

    '\n35. OCR Error from the User: uithTeipht goGv:men: as that Cotoany ds'
    '   The JSON output is: '
    '     Original text line: uithTeipht goGv:men: as that Cotoany ds'
    '     Corrected text line: with eight good men: as that Company is'
    '     Confidence (%): 78 '
    '     Justification: Corrected the substitutions "uithTeipht" to "with eight", "goGv:men:" to "good men:", '
    '"Cotoany" to "Company", and "ds" to "is". '

    '\n36. OCR Error from the User: fithout aaSer Aon, ifO ov will do thau dute, yn'
    '   The JSON output is: '
    '     Original text line: fithout aaSer Aon, ifO ov will do thau dute, yn'
    '     Corrected text line: without a Surgeon, if you will do that duty, an'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "fithout" to "without", "aaSer Aon" to "a Surgeon,", "ifO ov" '
    'to "if you", "thau dute" to "that duty", and "yn" to "an". '

    '\n37. OCR Error from the User: allowanwe5wicl bs dade nnu for it., ou arT'
    '   The JSON output is: '
    '     Original text line: allowanwe5wicl bs dade nnu for it., ou arT'
    '     Corrected text line: allowance will be made you for it. You are'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "allowanwe5wicl" to "allowance will", "bs dade nnu" to '
    '"be made you", "for it." to "for it.", "ou" to "You", and "arT" to "are". '

    '\n38. OCR Error from the User: tm phovrdo mkDi,cnes, Vc. upon tee bemt tsrms'
    '   The JSON output is: '
    '     Original text line: tm phovrdo mkDi,cnes, Vc. upon tee bemt tsrms'
    '     Corrected text line: to provide medicines, Vc. upon the best terms'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "tm" to "to", "phovrdo" to "provide", "mkDi,cnes" to '
    '"medicines,", "tee" to "the", and "bemt tsrms" to "best terms". '

    '\n39. OCR Error from the User: wouycmn. This OOder I -lcecS whll be .m.edi-'
    '   The JSON output is: '
    '     Original text line: wouycmn. This OOder I -lcecS whll be .m.edi-'
    '     Corrected text line: you can. This Order I expect will be immedi-'
    '     Confidence (%): 84 '
    '     Justification: Corrected the substitutions "wouycmn" to "you can", "OOder" to "Order", "I -lcecS" '
    'to "I expect", "whll" to "will", and ".m.edi-" to "immedi-". '

    '\n40. OCR Error from the User: afely domphiedBwith, tnd that rfcDAlayW be of-'
    '   The JSON output is: '
    '     Original text line: afely domphiedBwith, tnd that rfcDAlayW be of-'
    '     Corrected text line: ately complied with; and that no Delays be of-'
    '     Confidence (%): 80 '
    '     Justification: Corrected the substitutions "afely" to "ately", "domphiedBwith" to "complied with;", "tnd" '
    'to "and", "rfcDAlayW" to "no Delays", and "be of-" to "be of-". '

    '\n41. OCR Error from the User: fer-d. YYP a eMtohacio-nt wiuh Captain Belh'
    '   The JSON output is: '
    '     Original text line: fer-d. YYP a eMtohacio-nt wiuh Captain Belh'
    '     Corrected text line: fered. You are to account with Captain Bell'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "fer-d" to "fered.", "YYP" to "You", "a eMtohacio-nt" to '
    '"are to account", "wiuh" to "with", and "Belh" to "Bell". '

    '\n42. OCR Error from the User: foy youyMrkcruilTngdmovey, ,efore you letve hlS.'
    '   The JSON output is: '
    '     Original text line: foy youyMrkcruilTngdmovey, ,efore you letve hlS.'
    '     Corrected text line: for your recruiting money, before you leave him.'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "foy" to "for", "youyMrkcruilTngdmovey," to '
    '"your recruiting money,", ",efore" to "before", "letve" to "leave", and "hlS." to "him.". '

    '\n43. OCR Error from the User: hf dou .eould aOrive at lugusWa oourt HoudS'
    '   The JSON output is: '
    '     Original text line: hf dou .eould aOrive at lugusWa oourt HoudS'
    '     Corrected text line: If you should arrive at Augusta Court House'
    '     Confidence (%): 88 '
    '     Justification: Corrected the substitutions "hf" to "If", "dou" to "you", ".eould" to "should", '
    '"aOrive" to "arrive", "lugusWa" to "Augusta", "oourt" to "Court", and "HoudS" to "House". '

    '\n44. OCR Error from the User: lefork Selgednt Wimper anf uis Iarty, l-u agA to'
    '   The JSON output is: '
    '     Original text line: lefork Selgednt Wimper anf uis Iarty, l-u agA to'
    '     Corrected text line: before Sergeant Wilper and his Party, you are to'
    '     Confidence (%): 87 '
    '     Justification: Corrected the substitutions "lefork" to "before", "Selgednt" to "Sergeant", '
    '"Wimper" to "Wilper", "anf uis" to "and his", "Iarty," to "Party,", "l-u agA to" to "you are to". '

    '\n45. OCR Error from the User: haitMShereAcn il he soins, in order tfUesoorti7he'
    '   The JSON output is: '
    '     Original text line: haitMShereAcn il he soins, in order tfUesoorti7he'
    '     Corrected text line: halt there until he joins, in order to escort the'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "haitMShereAcn" to "halt there", "il he soins" to '
    '"until he joins", "tfUesoorti7he" to "to escort the". '

    '\n46. OCR Error from the User: Ammunitifn, Vcm fCr the F ,t; wIkteMyoP wall'
    '   The JSON output is: '
    '     Original text line: Ammunitifn, Vcm fCr the F ,t; wIkteMyoP wall'
    '     Corrected text line: Ammunition, Vc. for the Fort; where you will'
    '     Confidence (%): 84 '
    '     Justification: Corrected the substitutions "Ammunitifn" to "Ammunition,", "Vcm fCr" to "Vc. for", '
    '"the F ,t;" to "the Fort;", "wIkteMyoP" to "where you", and "wall" to "will". '

    '\n47. OCR Error from the User: r.ceive Rloymes -rd Armp fhr the hen.'
    '   The JSON output is: '
    '     Original text line: r.ceive Rloymes -rd Armp fhr the hen.'
    '     Corrected text line: receive Clothes and Arms for the Men.'
    '     Confidence (%): 88 '
    '     Justification: Corrected the substitutions "r.ceive" to "receive", "Rloymes" to "Clothes", "-rd" to "and", '
    '"Armp" to "Arms", and "fhr the hen." to "for the Men.". '

    '\n48. OCR Error from the User: 21tn. To Cap.ailMBesl of7tT- Vwrginia Reeiment.'
    '   The JSON output is: '
    '     Original text line: 21tn. To Cap.ailMBesl of7tT- Vwrginia Reeiment.'
    '     Corrected text line: 28th. To Captain Bell of the Virginia Regiment.'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "21tn." to "28th.", "Cap.ailMBesl" to "Captain Bell", '
    '"of7tT-" to "of the", and "Vwrginia Reeiment." to "Virginia Regiment.". '

    '\n49. OCR Error from the User: T have Orwgred mnsbgn FAkming'
    '   The JSON output is: '
    '     Original text line: T have Orwgred mnsbgn FAkming'
    '     Corrected text line: I have Ordered Ensign Fleming'
    '     Confidence (%): 83 '
    '     Justification: Corrected the substitutions "T" to "I", "Orwgred" to "Ordered", "mnsbgn" to '
    '"Ensign", and "FAkming" to "Fleming". '

    '\n50. OCR Error from the User: tI RepaisvtoSCaptainhHoggs Cemmany vithxei,Jt'
    '   The JSON output is: '
    '     Original text line: tI RepaisvtoSCaptainhHoggs Cemmany vithxei,Jt'
    '     Corrected text line: to Repair to Captain Hoggs Company with eight'
    '     Confidence (%): 85 '
    '     Justification: Corrected the substitutions "tI" to "to", "RepaisvtoS" to "Repair to", "CaptainhHoggs" to '
    '"Captain Hoggs", "Cemmany" to "Company", "vithxei,Jt" to "with eight". '
)
