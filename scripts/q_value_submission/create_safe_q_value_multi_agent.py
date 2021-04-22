import tensorflow as tf
import os
import base64
import bz2
import pickle
import sys
import argparse
from tqdm import tqdm

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    all_weights = []
    for model_path in tqdm(args.model_paths):
        model = tf.keras.models.load_model(model_path, compile=False)
        model = tf.keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)
        weight_base64 = base64.b64encode(bz2.compress(pickle.dumps(model.get_weights())))
        all_weights.append(weight_base64)

    template_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'safe_q_value_multi_agent_template.py')
    with open(template_path, 'r') as f:
        template_text = f.read()

    template_text = template_text.replace('paste_weights_here', str(all_weights))
    template_text = template_text.replace('paste_model_path_here', str(args.model_paths))
    with open(args.agent_path, 'w') as f:
        f.write(template_text)

def parse_args(args):
    epilog = """
    python create_safe_q_value_agent.py /mnt/hdd0/Kaggle/hungry_geese/models/37_playing_against_frozen_agents/05_continue_lr2e5/epoch_3520.h5 /mnt/hdd0/MEGA/AI/22 Kaggle/hungry_geese/data/agents/rhaegar.py
    """
    parser = argparse.ArgumentParser(
        description='Prints the weights in a format that is useful for making the submission',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('agent_path', help='Path to the python file that will be created')
    parser.add_argument('model_paths', help='Path to models', nargs='+')
    return parser.parse_args(args)

if __name__ == '__main__':
    main()