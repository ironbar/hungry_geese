import tensorflow as tf
import base64
import bz2
import pickle
import sys
import argparse

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model = tf.keras.models.Model(inputs=model.inputs[:2], outputs=model.layers[-3].output)
    weight_base64 = base64.b64encode(bz2.compress(pickle.dumps(model.get_weights())))
    print(weight_base64)

def parse_args(args):
    epilog = """
    """
    parser = argparse.ArgumentParser(
        description='Prints the weights in a format that is useful for making the submission',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('model_path', help='Path to model')
    return parser.parse_args(args)

if __name__ == '__main__':
    main()