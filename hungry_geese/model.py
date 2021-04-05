import tensorflow.keras as keras

def simple_model(conv_filters, conv_activations, mlp_units, mlp_activations):
    board_input, features_input = _create_model_input()

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


def _create_model_input():
    board_input = keras.layers.Input((7, 11, 17), name='board_input')
    features_input = keras.layers.Input((9,), name='features_input')
    return board_input, features_input


def create_model_for_training(model):
    input_mask = keras.layers.Input((4,), name='input_mask')
    output = keras.backend.sum(input_mask*model.output, axis=-1)
    new_model = keras.models.Model(inputs=(model.inputs + [input_mask]), outputs=output)
    return new_model


def torus_model(torus_filters, summary_conv_filters, summary_conv_activations,
                feature_encoder_units, feature_encoder_activation,
                mlp_units, mlp_activations):
    """
    The idea is that the torus blocks extract features from the board, then we have some convolutional
    layers to summarize those features, concatenate with hand crafted features and a final mlp
    """
    board_input, features_input = _create_model_input()

    board_encoder = board_input
    for n_filters in torus_filters:
        board_encoder = torus_conv_bn_relu_block(board_encoder, n_filters)

    for n_filters, activation in zip(summary_conv_filters, summary_conv_activations):
        board_encoder = conv_bn_activation_block(board_encoder, n_filters, activation)
    board_encoder = keras.layers.Flatten()(board_encoder)

    features_encoder = dense_bn_activation_block(
        features_input, feature_encoder_units, feature_encoder_activation)

    output = keras.layers.concatenate([board_encoder, features_encoder])
    for units, activation in zip(mlp_units, mlp_activations):
        output = dense_bn_activation_block(output, units, activation)

    output = keras.layers.Dense(4, activation='linear', name='action')(output)

    model = keras.models.Model(inputs=[board_input, features_input], outputs=output)
    return model


def torus_conv_bn_relu_block(x, n_filters):
    # import tensorflow.keras as keras
    # x = keras.layers.Lambda(
    #     lambda x: keras.backend.tile(x, n=(1, 3, 3, 1))[:, x.shape[1]-1:2*x.shape[1]+1, x.shape[2]-1:2*x.shape[2]+1,:],
    #     output_shape=lambda input_shape: (None, input_shape[1]+2, 3*input_shape[2]+2, input_shape[3]))(x)
    x = keras.backend.tile(x, n=(1, 3, 3, 1))[:, x.shape[1]-1:2*x.shape[1]+1, x.shape[2]-1:2*x.shape[2]+1,:]
    x = keras.layers.Conv2D(n_filters, (3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x

def conv_bn_activation_block(x, n_filters, activation):
    x = keras.layers.Conv2D(n_filters, (3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    return x

def dense_bn_activation_block(x, units, activation):
    x = keras.layers.Dense(units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    return x