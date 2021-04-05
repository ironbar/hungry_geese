import pytest
import tensorflow.keras as keras

from hungry_geese.model import simple_model, torus_model, create_model_for_training

def test_create_simple_model():
    model = simple_model(
        conv_filters=[1, 1, 1],
        conv_activations=['relu', 'relu', 'relu'],
        mlp_units=[1, 1],
        mlp_activations=['relu', 'tanh'])
    keras.backend.clear_session()

def test_create_model_for_training():
    model = simple_model(
        conv_filters=[1, 1, 1],
        conv_activations=['relu', 'relu', 'relu'],
        mlp_units=[1, 1],
        mlp_activations=['relu', 'tanh'])
    training_model = create_model_for_training(model)
    keras.backend.clear_session()

def test_create_torus_model():
    model = torus_model(
        torus_filters=[1, 1],
        summary_conv_filters=[1, 1, 1],
        summary_conv_activations=['relu', 'relu', 'relu'],
        feature_encoder_units=1,
        feature_encoder_activation='relu',
        mlp_units=[1, 1],
        mlp_activations=['relu', 'tanh'],
        )
    model.summary()
    keras.backend.clear_session()