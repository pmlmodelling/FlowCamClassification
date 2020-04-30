# common library imports
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        EarlyStopping, TensorBoard)
import numpy as np
import pandas as pd
from os import listdir, path
import pickle

from typing import Dict, List, Tuple

ROOT_PATH: path.abspath = path.abspath("./")


DEFAULT_BATCH_SIZE: int = 32
DEFAULT_BEST_MODELS: List[str] = [
    'cnn_vgg_4_modules_test.h5', 'cnn_inception_3_modules_test.h5']
# example from orignal application
# DEFAULT_BEST_MODELS = [
#     'cnn_inceptionres_7_modules_model.h5',
#     'cnn_residual_3_modules_model.h5',
#     'cnn_vgg_4_modules_model.h5']
DEFAULT_CONVNET_MODEL_TYPE: str = "Inception"
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (64, 101)
DEFAULT_MIN_SAMPLES: int = 14
DEFAULT_CNN_MODEL_NAME: str = "cnn_inception_3_modules_test.h5"
DEFAULT_MLP_MODEL_NAME: str = "mlp_test.h5"
DEFAULT_CONCAT_MODEL_NAME: str = "collab_test.h5"
DEFAULT_NUM_EPOCHS: int = 20
DEFAULT_NUM_MODULES: int = 2
DEFAULT_OUTPUT_DIR: path.join = path.join("./reports")
DEFAULT_PROCESSED_TRAINING_DATA_PATH: path.join = path.join(
    ROOT_PATH, "./processed_data", "plankton_data_101x64_final.pkl")
DEFAULT_TRAINED_MODELS_PATH: path.join = path.join("./trained_models")
DEFAULT_TRAINED_MODEL_PATH: path.join = \
    path.join(ROOT_PATH, "merged_model_1_model.h5")
DEFAULT_FLOWCAM_DATA_DIR: path.join = path.join(ROOT_PATH, "flowcam_data")


def get_image_augmentation_opt(batch_size: int) -> Dict:
    """ Gets passed to the image augmentation function, configures the
    ranges of various augmentations."""
    return {
        'ROT_RANGE': 37,
        'WIDTH_SHIFT_RANGE': 0.3,
        'HEIGHT_SHIFT_RANGE': 0.3,
        'SHEAR_RANGE': 10,
        'ZOOM_RANGE': 0.4,
        'HOR_FLIP': True,
        'VER_FLIP': True,
        'BATCH_SIZE': batch_size}


def create_training_data() -> None: 
    """ From the original FlowCam data, this function carries out
    the entire processing pipeline. Including extracting, rotating and
    resizing of images. Data cleanup, feature engineering, stnadardization etc.
    Before eventually splitting the data into training, testing and validation
    sets and storing them at the specified `processed_training_data_path`
    """
    from collections import defaultdict

    from data_preprocessing import (
        drop_columns, prepare_training_data, process_attributes)
    from flowcam_data_processor import FlowcamDataProcessor

    class_dict: defaultdict = defaultdict(dict)
    print("Preparing training data from raw FlowCam files, \
          this may take a while")
    # STAGE ONE - PROCESS RAW FLOWCAM DATA
    # TODO: This should be a function in data_processing.py
    dataframes: List = []
    all_images: List = []
    for idx, directory in enumerate(listdir(args.flowcam_data_dir)):
        # our class_dict will save human readable name to integer key
        class_dict[idx] = directory
        # set up the flowcam data processor
        fdp: FlowcamDataProcessor = FlowcamDataProcessor(
            path.join(ROOT_PATH, args.flowcam_data_dir, directory))
        # parse the flb files (in this case) and store in a dataframe
        df: pd.DataFrame = fdp.process_lst_or_flb_files(idx)
        # using the dataframe snip the images that corrospond to each row
        images: np.array = fdp.snip_images(df, args.desired_image_size)
        # there is one dataframe for each species
        dataframes.append(df)
        # one set of images for each species
        all_images.append(images)

    # join all species dataframes into one "master" dataframe
    df: pd.DataFrame = pd.concat(dataframes, sort=True)
    # store information for the entire dataset
    df.to_csv(path.join(ROOT_PATH, './processed_data', 'original_data.csv'))
    # free up memory
    del dataframes
    # similarly stack all images into one numpy matrix
    all_images: np.array = np.vstack(all_images)

    # STAGE TWO - DATA PREPROCESSING
    # This includes feature engineering, standardization and data splitting

    # deal with missing values / feature engineering
    df: pd.DataFrame = process_attributes(df)
    # drop features that are no longer needed for training
    df = drop_columns(df)
    # split into train/val/test and standardize features
    trainAttrX, valAttrX, testAttrX, trainImagesX, \
        valImagesX, testImagesX, y_train, y_val, y_test = \
        prepare_training_data(df, all_images, args.min_samples)

    # save the class dictionary, for later use,
    # each int key maps to string species value
    np.save(path.join(ROOT_PATH, './processed_data', 'class_dict.npy'),
            class_dict)
    # finally save the training data for future use
    with open(args.processed_training_data_path, "wb") as f:
        pickle.dump((trainAttrX, valAttrX, testAttrX, trainImagesX,
                     valImagesX, testImagesX, y_train, y_val, y_test),
                    f, protocol=4)


def evaluate_convnet_model(model_name: str) -> Tuple[float, float, float, int]:
    """Calculate various scores from a single ConvNet model

    Arguments:
        model_name {str} -- [description]

    Returns:
        Tuple[float, float, float, int] -- [description]
    """    

    from tensorflow.keras.models import load_model
    from utils import print_metric_scores

    with open(args.processed_training_data_path, "rb") as file:
        trainAttrX, valAttrX, testAttrX, trainImagesX, \
            valImagesX, testImagesX, y_train, y_val, y_test = pickle.load(file)

    convnet_model_path: path.join = \
        path.join(args.trained_models_path, model_name)
    convnet_model = load_model(convnet_model_path)
    num_params: int = convnet_model.count_params()
    print("Num params: ", num_params)
    y_pred_test = convnet_model.predict(testImagesX)
    y_pred_train = convnet_model.predict(trainImagesX)
    # output = tf.keras.metrics.top_k_categorical_accuracy(
    #     y_test, y_pred_test, k=3)

    # TODO: This needs adapting to work with TF 2.0, for now score 
    # is recorded as 0
    # with tf.Session() as sess: top_3_acc = sess.run(output)
    top_3_acc = 0

    y_t = [np.where(y == 1)[0][0] for y in y_test]
    y_p = [np.argmax(y) for y in y_pred_test]

    y_t_train = [np.where(y == 1)[0][0] for y in y_train]
    y_p_train = [np.argmax(y) for y in y_pred_train]

    # print("Top-3 accuracy: {}".format(top_3_acc))

    acc, precision, recall, f1 = print_metric_scores(y_t, y_p)

    # training accutacy:
    print("Training accuracy: ")
    acc_train, _, _, _ = print_metric_scores(y_t_train, y_p_train)

    return acc_train, acc, top_3_acc, f1, num_params


def evaluate_convnet_models() -> None:
    """Evaluate multiple convnet models and write results to csv.
    For this to work each stored ConvNet model filename should begin with
    'cnn' and have the .h5 extension."""

    import csv
    print('Evaluating')
    with open('experiments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'train_acc', 'test_acc',
                         'top_3_acc', 'f1_score', 'num_params'])
        models = listdir(args.trained_models_path)
        for model in models:
            if model.startswith("cnn") and model.endswith(".h5"):
                acc_train, acc, top_3_acc, f1, num_params = \
                    evaluate_convnet_model(model)
                writer.writerow(
                    [model, acc_train, acc, top_3_acc, f1, num_params])


def fully_evaluate_convnet_models() -> pd.DataFrame:
    """ Unlike the earlier function that only calculates overall scores,
     this gives a class by class breakdown
     and saves the results to a panda dataframe on disk
     """
    from tensorflow.keras.models import load_model
    from utils import print_metric_scores, dataframe_from_scores

    def get_predictions(y_pred: np.array, y_test: np.array) -> pd.DataFrame:

        y_t: List = [np.where(r == 1)[0][0] for r in y_test]
        y_p: List = [np.argmax(r) for r in y_pred]

        y_t: np.array = np.array(y_t)
        y_p: np.array = np.array(y_p)

        a, p, r, f1 = print_metric_scores(y_t, y_p, average=None)

        df: pd.DataFrame = dataframe_from_scores(a, p, r, f1, y_p)
        return df

    with open(args.processed_training_data_path, "rb") as file:
        trainAttrX, valAttrX, testAttrX, trainImagesX, \
            valImagesX, testImagesX, y_train, y_val, y_test = pickle.load(file)

    main_df: pd.DataFrame = pd.DataFrame()
    # load in the pretrained ConvNet models
    for i, model_name in enumerate(DEFAULT_BEST_MODELS):
        model = load_model(
            path.join(args.trained_models_path, model_name))
        y_pred = model.predict(testImagesX)
        y_t = [np.where(y == 1)[0][0] for y in y_test]
        y_p = [np.argmax(y) for y in y_pred]
        acc, precision, recall, f1 = print_metric_scores(
            y_t, y_p, average=None)
        vgg_df = get_predictions(y_pred, y_test)
        main_df = pd.concat([main_df, vgg_df], axis=1, sort=False)
    main_df.to_csv('./reports/cnn_scores.csv')
    print("A full report has been saved in reports directory")


def train_cnn(model_type: str, num_modules: int) -> None:
    """
    Given a model type this will create a ConvNet architecture with the
    number of modules specified. By default these are scaled up 
    depending on the original paper they are based on.

    model_type : str
        One of: "InceptionResidual", "Inception", "Residual"
                "VGG", "dense"
    """
    # TODO: Ensure model_type paramater is one of the five modeltypes
    # TODO: Ensure num_modules is a valid amount given the modeltype
    from keras_model import KerasModel
    from models import select_convnet_model
    from utils import create_augmented_images_generator

    with open(args.processed_training_data_path, "rb") as file:
        trainAttrX, valAttrX, testAttrX, trainImagesX, \
            valImagesX, testImagesX, y_train, y_val, y_test = pickle.load(file)

    model = select_convnet_model(y_train, model_type, num_modules)

    keras_model: KerasModel = KerasModel(model)

    train_gen_opt: Dict = get_image_augmentation_opt(args.batch_size)

    train_gen = create_augmented_images_generator(trainAttrX,
                                                  trainImagesX,
                                                  y_train,
                                                  train_gen_opt,
                                                  only_images=True)

    training_opt: Dict = \
        {'BATCH_SIZE': args.batch_size, 'NB_EPOCH': args.num_epochs}
    keras_model.compile_model()
    history = keras_model.train_model(
        train_gen,
        (valImagesX, y_val),
        args.cnn_model_name,
        y_train,
        training_opt=training_opt)

    with open(path.join(args.trained_models_path, '{}_history'.format(
            args.cnn_model_name)), 'wb') as file:
        pickle.dump(history.history, file)


def train_lightweight_collaborative_model() -> None:
    """ This is used when creating a collaborative model
    containing a single ConvNet and MLP
    TODO: This should be deprecated, and instead configure
    train_multi_model_collab to work with a single ConvNet
    """
    from keras_model import KerasModel
    from models import combine_two_models
    from utils import create_augmented_images_generator, load_trained_model

    with open(args.processed_training_data_path, "rb") as file:
        trainAttrX, valAttrX, testAttrX, trainImagesX, \
            valImagesX, testImagesX, y_train, y_val, y_test = pickle.load(file)

    mlpmodel = load_trained_model(
        path.join(args.trained_models_path, args.mlp_model_name))
    cnnmodel = load_trained_model(
        path.join(args.trained_models_path, args.cnn_model_name))

    # rename the mlp layer names to avoid conflicts
    for i, layer in enumerate(mlpmodel.layers):
        layer._name = 'layer_{}'.format(i)

    model = combine_two_models(cnnmodel, mlpmodel, y_train)
    keras_model = KerasModel(model)

    train_gen_opt: Dict = get_image_augmentation_opt(args.batch_size)

    val_gen_opt: Dict = {
        'ROT_RANGE': 0,
        'WIDTH_SHIFT_RANGE': 0.0,
        'HEIGHT_SHIFT_RANGE': 0.0,
        'SHEAR_RANGE': 0.0,
        'ZOOM_RANGE': 0.0,
        'HOR_FLIP': False,
        'VER_FLIP': False,
        'BATCH_SIZE': args.batch_size}

    train_gen = create_augmented_images_generator(
        trainAttrX, trainImagesX, y_train, train_gen_opt, only_images=False)
    val_gen = create_augmented_images_generator(
        valAttrX, valImagesX, y_val, val_gen_opt, only_images=False)

    training_opt: Dict = \
        {'BATCH_SIZE': args.batch_size, 'NB_EPOCH': args.num_epochs}
    keras_model.compile_model()
    history = keras_model.train_model(
        train_gen,
        val_gen,
        args.concat_model_name,
        y_train,
        training_opt=training_opt)

    with open(path.join(args.trained_models_path, '{}_history'.format(
            args.concat_model_name)), 'wb') as file:
        pickle.dump(history.history, file)


def train_mlp() -> None:
    """ Constructs the MLP model stated in the original paper,
    trains and stores a snapshot of the best model found in training"""
    from models import multilayer_perceptron
    from sklearn.utils import class_weight

    with open(args.processed_training_data_path, "rb") as file:
        trainAttrX, valAttrX, testAttrX, trainImagesX, \
            valImagesX, testImagesX, y_train, y_val, y_test = pickle.load(file)

    model = multilayer_perceptron(trainAttrX.shape[1], y_train.shape[1])

    output_path = path.join(args.trained_models_path, args.mlp_model_name)
    # save the best model so far when training
    checkpoint = ModelCheckpoint(
        output_path, monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max')
    # lower learning rate when models learning has plateaued
    lr_drop = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=50, min_lr=0.000001)
    # stop training if signs of model convergence
    early_stopping = EarlyStopping(monitor='loss', patience=80)
    # enables us to start tensorboard from console for diagnostic tools
    tensor_board = TensorBoard(
        log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, lr_drop, early_stopping, tensor_board]
    y = [np.where(r == 1)[0][0] for r in y_train]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y),
                                                      y)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(trainAttrX,
                        y_train,
                        epochs=200,
                        validation_data=(valAttrX, y_val),
                        callbacks=callbacks_list,
                        batch_size=5000,
                        class_weight=class_weights)

    with open(path.join(args.trained_models_path,
                        "{}_history".format(args.mlp_model_name)),
              'wb') as file:
        pickle.dump(history.history, file)


def train_multi_model_collab() -> None:
    """ Main function to create a collaborative model containing two or
    more ConvNets and an MLP model. The weights for each ConvNet and MLP
    are frozen. Instead a new concatanation layer to combine the outputs,
    a new fully connected layer and a fresh SoftMax layer is introduced to
    teach these models how to "collaborate".
    
    TODO: Add command line arguement to decide whether to inclue the
    MLP model. If anyone is reading this and want a ConvNet only collaboration
    simply comment out the part that loads the MLP model
    TODO: More appropiate layer names when they get renamed
    """
    from keras_model import KerasModel
    from models import combine_multiple_cnn_models
    from utils import create_augmented_images_generator, load_trained_model

    with open(args.processed_training_data_path, "rb") as file:
        trainAttrX, valAttrX, testAttrX, trainImagesX, \
            valImagesX, testImagesX, y_train, y_val, y_test = pickle.load(file)

    models: List = []

    # load in the pretrained ConvNet models
    for i, model_name in enumerate(DEFAULT_BEST_MODELS):
        model = load_trained_model(
            path.join(args.trained_models_path, model_name))
        for j, layer in enumerate(model.layers):
            layer._name = "model_{}_layer_{}".format(i, j)
        models.append(model)

    # load the trained mlp model seperately
    mlpmodel = load_trained_model(
        path.join(args.trained_models_path, args.mlp_model_name))
    for j, layer in enumerate(mlpmodel.layers):
        layer._name = "mlp_layer_{}".format(j)
    models.append(mlpmodel)

    # now combine these models into one
    model = combine_multiple_cnn_models(models, y_train)

    keras_model = KerasModel(model)
    keras_model.compile_model()

    train_gen_opt: Dict = get_image_augmentation_opt(args.batch_size)
    train_gen = create_augmented_images_generator(
        trainAttrX,
        trainImagesX,
        y_train,
        train_gen_opt,
        only_images=False,
        multiple_inputs=True)

    training_opt: Dict = \
        {'BATCH_SIZE': args.batch_size, 'NB_EPOCH': args.num_epochs}

    # TODO: Make input dynamic depending on the number of models passed
    # WARNING: This is hardcoded to two image and validation inputs - 
    # adjust if required
    history = keras_model.train_model(
        train_gen,
        ([valImagesX, valImagesX, valAttrX], y_val),
        args.concat_model_name,
        y_train,
        training_opt=training_opt)

    with open(path.join(args.trained_models_path, '{}_history'.format(
            args.concat_model_name)), 'wb') as file:
        pickle.dump(history.history, file)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("command",
                        metavar="<command>",
                        help="'create_training_data', 'train_mlp', \
                            'train_cnn', 'train_merge'")

    parser.add_argument('--output_dir', required=False,
                        default=DEFAULT_OUTPUT_DIR,
                        metavar="/path/to/output_dir/",
                        help='Directory of classified data')

    parser.add_argument('--model_path', required=False,
                        default=DEFAULT_TRAINED_MODEL_PATH,
                        metavar="/path/to/trained_model/",
                        help='Path to trained Keras model')

    parser.add_argument('--batch_size', required=False,
                        default=DEFAULT_BATCH_SIZE,
                        metavar="<BATCH SIZE>",
                        help='Number of samples to predict at any one time \
                            (adjust based on RAM available) Default: 32')

    parser.add_argument('--image_size', required=False,
                        default=DEFAULT_IMAGE_SIZE,
                        metavar="<IMAGE SIZE>",
                        help='What size each plankton sample should be \
                            resized to, should be same size as used \
                            during training')

    parser.add_argument('--flowcam_data_dir', required=False,
                        default=DEFAULT_FLOWCAM_DATA_DIR,
                        metavar="/path/to/training_data/dir",
                        help="""Training data, each species should be in \
                            its own directory,with each directory \
                            containing one or more tif image collages and \
                            one or more flb files""")

    parser.add_argument('--desired_image_size', required=False,
                        default=DEFAULT_IMAGE_SIZE,
                        metavar="<(w, h)>",
                        help='Resized image size')

    parser.add_argument('--min_samples', required=False,
                        default=DEFAULT_MIN_SAMPLES,
                        metavar="<13>",
                        help='Minimum amount of samples to keep for each\
                             particle in training data')

    parser.add_argument('--processed_training_data_path', required=False,
                        default=DEFAULT_PROCESSED_TRAINING_DATA_PATH,
                        metavar="/path/to/processed/trainingdata",
                        help="""Before training or evaluating model, \
                            training data is required.
            Run with command create_training_data first if required""")

    parser.add_argument('--trained_models_path', required=False,
                        default=DEFAULT_TRAINED_MODELS_PATH,
                        metavar="/path/to/trained/mlp_path",
                        help='Where should Keras save the best \
                            models during training')

    parser.add_argument('--mlp_model_name', required=False,
                        default=DEFAULT_MLP_MODEL_NAME,
                        metavar="mlp_model_name.h5",
                        help='filename of the mlp model')

    parser.add_argument('--cnn_model_name', required=False,
                        default=DEFAULT_CNN_MODEL_NAME,
                        metavar="cnn_model_name.h5",
                        help='filename of the cnn model')

    parser.add_argument('--num_epochs', required=False,
                        default=DEFAULT_NUM_EPOCHS,
                        metavar="<int>",
                        help='Number of epochs to use during training')

    parser.add_argument('--concat_model_name', required=False,
                        default=DEFAULT_CONCAT_MODEL_NAME,
                        metavar="<string>",
                        help='Filename of concatenated model')

    parser.add_argument('--convnet_model_type', required=False,
                        default=DEFAULT_CONVNET_MODEL_TYPE,
                        metavar="<string>",
                        help='One of: InceptionResidual, Inception, \
                            Residual, VGG or Dense')

    parser.add_argument('--num_modules', required=False,
                        default=DEFAULT_NUM_MODULES,
                        metavar="<int>",
                        help='Number of modules to use in selected \
                            ConvNet arcitecture')

    args = parser.parse_args()

    # TODO: Validate arguments
    # Example:
    # if args.command == "classify":
    #     assert args.flowcam_files,
    # "Argument --flowcam_files directory is required"

    if args.command == "create_training_data":
        create_training_data()

    if args.command == "evaluate_convnet_model":
        evaluate_convnet_model(args.cnn_model_name)

    if args.command == "fully_evaluate_convnet_models":
        fully_evaluate_convnet_models()

    if args.command == "evaluate_convnet_models":
        evaluate_convnet_models()

    if args.command == "train_cnn":
        train_cnn(args.convnet_model_type, args.num_modules)

    if args.command == "train_mlp":
        train_mlp()

    if args.command == "train_lw_collab_model":
        train_lightweight_collaborative_model()

    if args.command == "train_multi_cnn_colab_model":
        train_multi_model_collab()
