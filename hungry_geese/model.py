import tensorflow.keras as keras

def simple_model():
    # TODO: parametric model
    board_input = keras.layers.Input((7, 11, 17), name='board_input')
    features_input = keras.layers.Input((9,), name='features_input')

    board_encoder = board_input
    for _ in range(3):
        board_encoder = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='valid')(board_encoder)
    board_encoder = keras.layers.Flatten()(board_encoder)

    mlp_input = keras.layers.concatenate([board_encoder, features_input])
    output = keras.layers.Dense(16, activation='relu')(mlp_input)
    output = keras.layers.Dense(16, activation='tanh')(output)
    output = keras.layers.Dense(4, activation='linear', name='action')(output)

    model = keras.models.Model(inputs=[board_input, features_input], outputs=output)
    return model

def create_model_for_training(model):
    input_mask = keras.layers.Input((4,), name='input_mask')
    output = keras.backend.sum(input_mask*model.output, axis=-1)
    new_model = keras.models.Model(inputs=(model.inputs + [input_mask]), outputs=output)
    return new_model