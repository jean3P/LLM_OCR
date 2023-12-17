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

# Specify the directory to save the model
model_save_path = os.path.join(outputs_path, 'model', 'trained_trocr_model')
processor_save_path = os.path.join(outputs_path, 'model', 'trocr_processor')


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