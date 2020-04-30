from tensorflow.keras.initializers import glorot_uniform, Constant
from tensorflow.python.keras.layers import (Input, add, concatenate,
                                            BatchNormalization,
                                            AveragePooling2D,
                                            GlobalAveragePooling2D,
                                            MaxPool2D)
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          MaxPooling2D)
from tensorflow.python.keras.layers.core import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import numpy as np
from typing import List, Tuple

kernel_init = glorot_uniform()
bias_init = Constant(value=0.1)
regularizer = regularizers.l2(0.0003)
IMAGE_SIZE = (101, 64, 3)


def select_convnet_model(y_train: np.array,
                         model_type: str,
                         num_modules: int,
                         dropout_rate: float = 0.0) -> Model:
    """[summary]

    Arguments:
        y_train {np.array} -- [description]
        model_type {str} -- [description]
        num_modules {int} -- [description]

    Keyword Arguments:
        dropout_rate {float} -- [description] (default: {0.0})

    Raises:
        SystemExit: [description]

    Returns:
        Model -- [description]
    """
    from models import (
        cnn_dense_inspired,
        cnn_inception_inspired,
        cnn_residual_inspired,
        cnn_vgg_inspired)

    if model_type == "InceptionResidual":
        model = cnn_inception_inspired(
            num_classes=y_train.shape[1],
            input_shape=IMAGE_SIZE,
            add_residual=True,
            dropout_rate=dropout_rate,
            num_modules=num_modules)
        return model

    elif model_type == "Inception":
        model = cnn_inception_inspired(
            num_classes=y_train.shape[1],
            input_shape=IMAGE_SIZE,
            add_residual=False,
            dropout_rate=dropout_rate,
            num_modules=num_modules)
        return model

    elif model_type == "Residual":
        model = cnn_residual_inspired(
            num_classes=y_train.shape[1],
            input_shape=IMAGE_SIZE,
            dropout_rate=dropout_rate,
            num_modules=num_modules)
        return model

    elif model_type == "VGG":
        model = cnn_vgg_inspired(
            num_classes=y_train.shape[1],
            input_shape=IMAGE_SIZE,
            dropout_rate=dropout_rate,
            num_modules=num_modules)
        return model

    elif model_type == "Dense":
        model = cnn_dense_inspired(
            num_classes=y_train.shape[1],
            input_shape=IMAGE_SIZE,
            dropout_rate=dropout_rate,
            num_modules=num_modules)
        return model

    else:
        print("ConvNet model type not recognised")
        raise SystemExit


def combine_two_models(model1: Model,
                       model2: Model,
                       y: np.array,
                       dropout_rate: float = 0.5) -> Model:
    """Combines two models into one model that accepts two inputs
    and produces one output

    Arguments:
        model1 {Model} -- [description]
        model2 {Model} -- [description]
        y {np.array} -- [description]

    Keyword Arguments:
        dropout_rate {float} -- [description] (default: {0.5})

    Returns:
        Model -- [description]
    """

    combinedInput = concatenate(
        [model1.output, model2.output], name="mlp_cnn_concat")

    x = Dense(1024,
              activation='relu',
              kernel_initializer=kernel_init,
              bias_initializer=bias_init)(combinedInput)
    x = BatchNormalization(name='combined_bn')(x)

    # prediction layer
    x = Dense(y.shape[1], activation='softmax', name='combined_pred')(x)

    # images for model1
    # and model2 will accept categorical/numerical data
    model = Model(inputs=[model1.input, model2.input], outputs=x)

    return model


def combine_multiple_cnn_models(models: List[Model],
                                y: np.array,
                                dropout_rate: float = 0.5) -> Model:
    """Despite its name, this will combine multiple deep learning models
    (each individual should have their SoftMax layer removed)
    into a single model, by providing a concatanation layer and a fully
    connected layer

    TODO: rename this function

    Arguments:
        models {List[Model]} -- [description]
        y {np.array} -- [description]

    Keyword Arguments:
        dropout_rate {float} -- [description] (default: {0.5})

    Returns:
        Model -- [description]
    """
    combinedInput = concatenate(
        [model.output for model in models], name="cnn_concat")

    x = Dense(1024,
              activation='relu',
              kernel_initializer=kernel_init,
              bias_initializer=bias_init)(combinedInput)
    x = BatchNormalization(name='combined_bn')(x)

    x = Dense(y.shape[1], activation='softmax', name='combined_pred')(x)

    model = Model(inputs=[model.input for model in models], outputs=x)

    return model


def conv2d_bn_layer(
        layer_in: Model, num_filters: int, filter_size: int) -> Model:
    """A standard Convolution -> Activation -> Batch Normalisation layer

    Arguments:
        layer_in {Model} -- [description]
        num_filters {int} -- [description]
        filter_size {int} -- [description]

    Returns:
        Model -- [description]
    """
    layer_out = Conv2D(num_filters, (filter_size),
                       padding='same',
                       activation='relu',
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init,
                       kernel_regularizer=regularizer)(layer_in)

    layer_out = BatchNormalization(axis=3)(layer_out)

    return layer_out


def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out,
                     add_residual=False) -> Model:
    """A single Inception module whereby each parameter can be adjusted

    Szegedy, C., W. Liu, Y. Jia, and others. 2015.
    Going deeper with convolutions. Proceedings of the IEEE Computer
    Society Conference on Computer Vision and Pattern Recognition. 1–9.

    TODO: type innotations, and rename attributes

    Arguments:
        layer_in {[type]} -- [description]
        f1 {[type]} -- [description]
        f2_in {[type]} -- [description]
        f2_out {[type]} -- [description]
        f3_in {[type]} -- [description]
        f3_out {[type]} -- [description]
        f4_out {[type]} -- [description]

    Keyword Arguments:
        add_residual {bool} -- [description] (default: {False})

    Returns:
        Model -- [description]
    """
    merge_input = layer_in

    # check if the number of filters needs to be increase
    if layer_in.shape[-1] != f2_out:
        merge_input = Conv2D(f1+f2_out+f3_out+f4_out, (1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init)(layer_in)

    # 1x1 conv
    conv1 = conv2d_bn_layer(layer_in, f1, (1, 1))

    # 3x3 conv
    conv3 = conv2d_bn_layer(layer_in, f2_in, (1, 1))
    conv3 = conv2d_bn_layer(conv3, f2_out, (3, 3))

    # 5x5 conv
    conv5 = conv2d_bn_layer(layer_in, f3_in, (1, 1))
    conv5 = conv2d_bn_layer(conv5, f3_out, (5, 5))

    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = conv2d_bn_layer(pool, f4_out, (1, 1))

    # concatenate filters
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)

    if add_residual:
        layer_out = add([layer_out, merge_input])
        layer_out = Activation('relu')(layer_out)

    return layer_out


def dense_layer(layer_in: Model, num_filters: int) -> Model:
    """
    Similar to conv2d_bn_layer function earlier, but authors suggests
    different order of operations
    """
    x = BatchNormalization(axis=3)(layer_in)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3),
               padding="same",
               use_bias=False,
               kernel_initializer=kernel_init,
               kernel_regularizer=regularizer)(x)
    return x


def dense_module(layer_in: Model, num_layers: int, num_filters: int) -> Model:
    """Dense module

    Huang, G., Z. Liu, L. Van Der Maaten, and K. Q. Weinberger. 2017.
    Densely connected convolutional networks. Proceedings - 30th IEEE
    Conference on Computer Vision and Pattern Recognition, 
    CVPR 2017. 2261–2269.

    Arguments:
        layer_in {Model} -- [description]
        num_layers {int} -- [description]
        num_filters {int} -- [description]

    Returns:
        Model -- [description]
    """
    x = layer_in

    concat_feature = x
    for _ in range(num_layers):
        x = dense_layer(concat_feature, num_filters)
        concat_feature = concatenate([concat_feature, x], axis=-1)

    return x


def transition_module(layer_in: Model, num_filters: int) -> Model:
    """Used inbetween some layers in dense inspired models

    Arguments:
        layer_in {Model} -- [description]
        num_filters {int} -- [description]

    Returns:
        Model -- [description]
    """
    x = BatchNormalization(axis=3)(layer_in)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (1, 1),
               padding="same",
               use_bias=False,
               kernel_initializer=kernel_init,
               kernel_regularizer=regularizer)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def cnn_dense_inspired(num_classes: int,
                       input_shape: Tuple[int, int, int],
                       dropout_rate: float,
                       num_modules: int) -> Model:
    """[summary]

    Arguments:
        num_classes {int} -- [description]
        input_shape {Tuple[int, int, int]} -- [description]
        dropout_rate {float} -- [description]
        num_modules {int} -- [description]

    Returns:
        Model -- [description]
    """
    visible = Input(shape=input_shape)
    layer = Conv2D(64, (7, 7),
                   padding='same',
                   strides=(2, 2),
                   activation='relu',
                   name='conv_1_7x7/2',
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   kernel_regularizer=regularizer)(visible)

    x = MaxPool2D((2, 2), padding='same', strides=(
        2, 2), name='max_pool_1_2x2/2')(layer)

    # TODO: More efficent way to do this, But at least its clear
    # Each module has dense and transition unless its last module
    if num_modules == 1:
        x = dense_module(x, num_layers=6, num_filters=12)

    elif num_modules == 2:
        x = dense_module(x, num_layers=6, num_filters=12)
        x = transition_module(x, num_filters=12)
        x = dense_module(x, num_layers=12, num_filters=16)

    elif num_modules == 3:
        x = dense_module(x, num_layers=6, num_filters=12)
        x = transition_module(x, num_filters=12)
        x = dense_module(x, num_layers=12, num_filters=16)
        x = transition_module(x, num_filters=16)
        x = dense_module(x, num_layers=24, num_filters=20)

    elif num_modules == 4:
        x = dense_module(x, num_layers=6, num_filters=12)
        x = transition_module(x, num_filters=12)
        x = dense_module(x, num_layers=12, num_filters=16)
        x = transition_module(x, num_filters=16)
        x = dense_module(x, num_layers=24, num_filters=20)
        x = transition_module(x, num_filters=20)
        x = dense_module(x, num_layers=16, num_filters=24)

    elif num_modules >= 5:
        x = dense_module(x, num_layers=6, num_filters=12)
        x = transition_module(x, num_filters=12)
        x = dense_module(x, num_layers=12, num_filters=16)
        x = transition_module(x, num_filters=16)
        x = dense_module(x, num_layers=24, num_filters=20)
        x = transition_module(x, num_filters=20)
        x = dense_module(x, num_layers=16, num_filters=24)
        x = transition_module(x, num_filters=24)
        x = dense_module(x, num_layers=16, num_filters=28)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=visible, outputs=x)

    return model


def vgg_module(layer_in: Model,
               n_filters: int,
               dropout_rate: float,
               n_conv: int = 2) -> Model:
    """A VGG module, simply contains multiple 
    conv->activation->batch normalisation layers

    Simonyan, K., and A. Zisserman. 2015. Very deep convolutional networks 
    for large-scale image recognition. 3rd Int. Conf. Learn. Represent. 
    ICLR 2015 - Conf. Track Proc.

    Arguments:
        layer_in {Model} -- [description]
        n_filters {int} -- [description]
        dropout_rate {float} -- [description]

    Keyword Arguments:
        n_conv {int} -- [description] (default: {2})

    Returns:
        Model -- [description]
    """
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = conv2d_bn_layer(layer_in, n_filters, (3, 3))
        # add max pooling layer
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
    return layer_in


def cnn_inception_inspired(num_classes: int,
                           input_shape: Tuple[int, int, int],
                           add_residual: bool,
                           dropout_rate: float,
                           num_modules: int = 1) -> Model:
    """Creates an Inception (GoogLeNet) inspired model
    with optional residua connections between modules.

    Arguments:
        num_classes {int} -- [description]
        input_shape {Tuple[int, int, int]} -- [description]
        add_residual {bool} -- [description]
        dropout_rate {float} -- [description]

    Keyword Arguments:
        num_modules {int} -- [description] (default: {1})

    Returns:
        [Model] -- [description]
    """
    visible = Input(shape=input_shape)

    layer = inception_model_stem(visible)

    # inception block 1
    layer = inception_module(layer, 64, 96, 128, 16, 32, 32, add_residual)

    # conditional layers - smarter ways of achieving this
    # implemented in an understandable way
    if num_modules >= 2:
        layer = inception_module(layer, 128, 128, 192,
                                 32, 96, 64, add_residual)

    if num_modules >= 3:
        layer = inception_module(layer, 192, 96, 208, 16, 48, 64, add_residual)

    if num_modules >= 4:
        layer = inception_module(layer, 160, 112, 224,
                                 24, 64, 64, add_residual)

    if num_modules >= 5:
        layer = inception_module(layer, 128, 128, 256,
                                 24, 64, 64, add_residual)

    if num_modules >= 6:
        layer = inception_module(layer, 112, 144, 288,
                                 32, 64, 64, add_residual)

    if num_modules >= 7:
        layer = inception_module(layer, 256, 160, 320,
                                 32, 128, 128, add_residual)

    if num_modules >= 8:
        layer = inception_module(layer, 256, 160, 320,
                                 32, 128, 128, add_residual)

    # TODO: Should throw an exception if more than 8
    # Any number higher than 8 will build the 8 model version

    # output layers
    layer = GlobalAveragePooling2D(name='avg_pool')(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(num_classes, activation="softmax")(layer)

    model = Model(inputs=visible, outputs=layer)

    return model


def inception_model_stem(layer: Model) -> Model:
    """Implemented as per original paper, these layers occur
    directly after the input layer

    Arguments:
        layer {Model} -- [description]

    Returns:
        Model -- [description]
    """
    layer = Conv2D(64, (7, 7), padding='same', strides=(2, 2),
                   activation='relu', name='conv_1_7x7/2',
                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                   kernel_regularizer=regularizer)(layer)
    layer = MaxPool2D((3, 3), padding='same', strides=(2, 2),
                      name='max_pool_1_3x3/2')(layer)
    layer = Conv2D(64, (1, 1), padding='same', strides=(1, 1),
                   activation='relu', name='conv_2a_3x3/1',
                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                   kernel_regularizer=regularizer)(layer)
    layer = Conv2D(192, (3, 3), padding='same', strides=(1, 1),
                   activation='relu', name='conv_2b_3x3/1',
                   kernel_initializer=kernel_init, bias_initializer=bias_init,
                   kernel_regularizer=regularizer)(layer)
    layer = MaxPool2D((3, 3), padding='same', strides=(2, 2),
                      name='max_pool_2_3x3/2')(layer)
    return layer


def residual_module(layer_in: Model, n_filters: int) -> Model:
    """ Creates a residual module

    He, K., X. Zhang, S. Ren, and J. Sun. 2016. 
    Deep residual learning for image recognition. 
    Proceedings of the IEEE Computer Society Conference on 
    Computer Vision and Pattern Recognition. 770–778.

    Arguments:
        layer_in {Model} -- [description]
        n_filters {int} -- [description]

    Returns:
        Model -- [description]
    """
    merge_input = layer_in

    # check if the number of filters needs to increase
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1, 1), padding='same',
                             activation='relu', kernel_initializer=kernel_init,
                             bias_initializer=bias_init)(layer_in)

    # conv1
    conv1 = conv2d_bn_layer(layer_in, n_filters, (3, 3))
    # conv2
    conv2 = conv2d_bn_layer(conv1, n_filters, (3, 3))
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)

    return layer_out


def cnn_residual_inspired(num_classes, input_shape, dropout_rate, num_modules):
    visible = Input(shape=input_shape)

    layer = conv2d_bn_layer(visible, 64, (7, 7))

    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

    layer = residual_module(layer, 64)

    if num_modules >= 2:
        layer = residual_module(layer, 128)

    if num_modules >= 3:
        layer = residual_module(layer, 256)

    if num_modules >= 4:
        layer = residual_module(layer, 512)

    if num_modules >= 5:
        layer = residual_module(layer, 1024)

    if num_modules >= 6:
        layer = residual_module(layer, 2048)

    # output layers
    layer = GlobalAveragePooling2D(name='avg_pool')(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(num_classes, activation="softmax")(layer)

    model = Model(inputs=visible, outputs=layer)

    return model


def cnn_vgg_inspired(num_classes: int,
                     input_shape: Tuple[int, int, int],
                     dropout_rate: float,
                     num_modules: int) -> Model:
    """Creates a VGG inspired model

    Arguments:
        num_classes {int} -- [description]
        input_shape {Tuple[int, int, int]} -- [description]
        dropout_rate {float} -- [description]
        num_modules {int} -- [description]

    Returns:
        Model -- [description]
    """
    # define model input
    visible = Input(shape=input_shape)

    layer = vgg_module(visible, 32, 2)

    if num_modules >= 2:
        layer = vgg_module(layer, 64, 2)

    if num_modules >= 3:
        layer = vgg_module(layer, 128, 2)

    if num_modules >= 4:
        layer = vgg_module(layer, 256, 2)

    if num_modules >= 5:
        layer = vgg_module(layer, 512, 2)

    # output layers
    layer = GlobalAveragePooling2D(name='avg_pool')(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(num_classes, activation="softmax")(layer)

    model = Model(inputs=visible, outputs=layer)

    return model


def multilayer_perceptron(dim: int,
                          num_classes: int,
                          activation: str = "relu") -> Model:
    """[summary]

    Arguments:
        dim {int} -- [description]
        num_classes {int} -- [description]

    Keyword Arguments:
        activation {str} -- [description] (default: {"relu"})

    Returns:
        Model -- [description]
    """
    model = Sequential()

    # input layers
    model.add(Dense(1024,
                    input_dim=dim,
                    activation=activation,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init))
    model.add(BatchNormalization())

    # hidden layers (1 in this case)
    for _ in range(1):
        model.add(Dense(512,
                        activation=activation,
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init))
        model.add(BatchNormalization())

    model.add(Dropout(0.62))

    # output layer
    model.add(Dense(num_classes, activation="softmax"))

    return model
