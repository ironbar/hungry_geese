import os
import sys
import argparse

REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_ENDING = """

###############################################################################
# Ending code
###############################################################################

import pickle
import bz2
import base64
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_reward(*args, **kwargs):
    return 0
get_cumulative_reward = get_reward


model = simple_model(
    conv_filters=[128, 128, 128, 128],
    conv_activations=['relu', 'relu', 'relu', 'relu'],
    mlp_units=[128, 128],
    mlp_activations=['relu', 'tanh'])

def get_weights():
    weights_b64 = paste_weights_here

    return weights_b64

###############################################################################
# paste_model_path_here
# paste_model_score_here
###############################################################################

weights_b64 = get_weights()
model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weights_b64))))
q_value_agent = QValueSafeAgent(model)

def agent(obs, config):
    return q_value_agent(obs, config)
"""

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    code = gather_code_from_modules()

    code = code + DEFAULT_ENDING
    with open('safe_q_value_agent_template.py', 'w') as f:
        f.write(code)

    code = code.replace(
        'q_value_agent = QValueSafeAgent(model)',
        'q_value_agent = QValueSafeAgentDataAugmentation(model)')
    with open('safe_q_value_agent_data_augmentation_template.py', 'w') as f:
        f.write(code)

def gather_code_from_modules():
    filepaths = [
        os.path.join(REPO_PATH, 'hungry_geese/utils.py'),
        os.path.join(REPO_PATH, 'hungry_geese/definitions.py'),
        os.path.join(REPO_PATH, 'hungry_geese/state.py'),
        os.path.join(REPO_PATH, 'hungry_geese/heuristic.py'),
        os.path.join(REPO_PATH, 'hungry_geese/actions.py'),
        os.path.join(REPO_PATH, 'hungry_geese/agents/q_value.py'),
        os.path.join(REPO_PATH, 'hungry_geese/model.py'),
    ]
    code = ''
    for filepath in filepaths:
        module = get_clean_module(filepath)
        code += '\n\n\n"""\n'
        code += os.path.basename(filepath)
        code += '\n"""\n\n'
        code += module
    return code

def get_clean_module(filepath):
    with open(filepath, 'r') as f:
        module = f.read()
    lines = module.split('\n')
    lines = [line for line in lines if not line.startswith('from hungry_geese')]
    if filepath.endswith('definitions.py'):
        lines = lines[:-5]
    module = '\n'.join(lines)
    return module

def parse_args(args):
    epilog = """
    """
    description = """
    Create agent template for later making submissions
    """
    parser = argparse.ArgumentParser(description=description, epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    return parser.parse_args(args)

if __name__ == '__main__':
    main()