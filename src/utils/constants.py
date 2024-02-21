import os
import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)
resources_path = os.path.join('./../resources')
washington_path = os.path.join(resources_path, 'washingtondb-v1.0')
outputs_path = os.path.join(resources_path, 'outputs')
transcription_path = os.path.join(washington_path, 'ground_truth', 'transcription.txt')
results_test_trocr = os.path.join(outputs_path, 'results_test')
results_LLM_mistral = os.path.join(outputs_path, 'results_LLM')
results_LLM_mistral_1 = os.path.join(outputs_path, 'results_LLM_Mistral7B_1')
results_LLM_mistral_2 = os.path.join(outputs_path, 'results_LLM_Mistral7B_1_v2')
results_LLM_mistral_3 = os.path.join(outputs_path, 'results_LLM_Mistral7B_1_v3')
# Specify the directory to save the model
model_save_path = os.path.join(outputs_path, 'model', 'trained_trocr_model')
processor_save_path = os.path.join(outputs_path, 'model', 'trocr_processor')

model_save_path_seq = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq')
model_save_path_seq_v2 = os.path.join(outputs_path, 'model', 'trained_trocr_model_seq_v2')
processor_save_path_seq = os.path.join(outputs_path, 'model', 'trocr_processor_seq')
processor_save_path_seq_v2 = os.path.join(outputs_path, 'model', 'trocr_processor_seq_v2')

REPLACEMENTS = {
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
