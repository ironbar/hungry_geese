import tensorflow.keras as keras

def simple_model(conv_filters, conv_activations, mlp_units, mlp_activations):
    # TODO: parametric model
    board_input = keras.layers.Input((7, 11, 17), name='board_input')
    features_input = keras.layers.Input((9,), name='features_input')

    board_encoder = board_input
    for n_filters, activation in zip(conv_filters, conv_activations):
        board_encoder = keras.layers.Conv2D(n_filters, (3, 3), activation=activation, padding='valid')(board_encoder)
    board_encoder = keras.layers.Flatten()(board_encoder)

    output = keras.layers.concatenate([board_encoder, features_input])
    for units, activation in zip(mlp_units, mlp_activations):
        output = keras.layers.Dense(units, activation=activation)(output)
    output = keras.layers.Dense(4, activation='linear', name='action')(output)

    model = keras.models.Model(inputs=[board_input, features_input], outputs=output)
    return model

def create_model_for_training(model):
    input_mask = keras.layers.Input((4,), name='input_mask')
    output = keras.backend.sum(input_mask*model.output, axis=-1)
    new_model = keras.models.Model(inputs=(model.inputs + [input_mask]), outputs=output)
    return new_model