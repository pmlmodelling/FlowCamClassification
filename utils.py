from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from seaborn import heatmap
from sklearn.externals.joblib import dump
# to load the scaler model for later: sc=load('std_scaler.bin')
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

from typing import Dict, List, Tuple


def create_augmented_images_generator(X_attributes: np.array,
                                      X_images: np.array,
                                      Y: np.array,
                                      opt: Dict = {},
                                      only_images: bool = False,
                                      multiple_inputs: bool = False) -> Tuple:
    """    Generates augmented images on the fly, will also yield the
    corrosponding attributes of the original image

    Arguments:
        X_attributes {np.array} -- [description]
        X_images {np.array} -- [description]
        Y {np.array} -- [description]

    Keyword Arguments:
        opt {Dict} -- image augmentation options (default: {{}})
        only_images {bool} -- can optionally not yield attributes
         (default: {False})
        multiple_inputs {bool} -- [description] (default: {False})


    Yields:
        Iterator[Tuple] -- [description]
    """

    # load image augmentation parameters
    ROT_RANGE = opt.get('ROT_RANGE', 90)
    WIDTH_SHIFT_RANGE = opt.get('WIDTH_SHIFT_RANGE', 0.3)
    HEIGHT_SHIFT_RANGE = opt.get('HEIGHT_SHIFT_RANGE', 0.3)
    SHEAR_RANGE = opt.get('SHEAR_RANGE', 10)
    ZOOM_RANGE = opt.get('ZOOM_RANGE', 0.3)
    HOR_FLIP = opt.get('HOR_FLIP', True)
    VER_FLIP = opt.get('VER_FLIP', True)
    BATCH_SIZE = opt.get('BATCH_SIZE', 32)

    num_samples: int = X_images.shape[0]

    while True:
        # create image generator
        datagen = ImageDataGenerator(
            rotation_range=ROT_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            shear_range=SHEAR_RANGE,
            zoom_range=ZOOM_RANGE,
            horizontal_flip=HOR_FLIP,
            vertical_flip=VER_FLIP,
            fill_mode='nearest')
        # shuffled indices, don't want potential model to learn order of data
        idx: int = np.random.permutation(num_samples)

        # where the augmentation happens
        batches = datagen.flow(
            X_images[idx], Y[idx], batch_size=BATCH_SIZE, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            # TODO: Multiple inputs assumes three images / one attribute
            # Multiple inputs refers to multiple image inputs
            # This is used for training a collab model
            # This could be made to be more flexible
            if multiple_inputs:
                if only_images:
                    # returns 3 images and the labels
                    yield [batch[0], batch[0]], batch[1]
                else:
                    # returns 3 images, attributes and the labels
                    yield [batch[0],
                           batch[0],
                           X_attributes[idx[idx0:idx1]]], batch[1]
            # Used for training a single ConvNet
            # also for the light weight collab model
            else:
                if only_images:
                    # only one set of images and the labels
                    yield [batch[0]], batch[1]
                else:
                    # one set of images and the attributes
                    yield [batch[0], X_attributes[idx[idx0:idx1]]], batch[1]

            idx0 = idx1
            if idx1 >= num_samples:
                break


def load_trained_model(path_to_model: path, trainable: bool = False) -> Model:
    """ Loads a snapshot of a model and removes the output layer.
    Will optionally freeze the model's weights.

    Arguments:
        path_to_model {path} -- file path to a previously trained model

    Keyword Arguments:
        trainable {bool} -- Should the weights be frozen (default: {False})

    Returns:
        {Model} -- TensorFlow (Keras) model with the output removed
    """
    from tensorflow.keras import Model
    from tensorflow.keras.models import load_model

    model: Model = load_model(path_to_model)
    model = Model(model.input, model.layers[-2].output)
    for l in model.layers:
        l.trainable = trainable
    return model


def standardize(Xtrain: np.array,
                Xval: np.array,
                Xtest: np.array) -> Tuple[np.array, np.array, np.array]:
    """Standardise features (mean of 0 and std of 1)

    Arguments:
        Xtrain {np.array} -- [description]
        Xval {np.array} -- [description]
        Xtest {np.array} -- [description]

    Returns:
        Tuple[np.array, np.array, np.array] -- [description]
    """

    rescaler = StandardScaler()

    Xtrain = rescaler.fit_transform(Xtrain.astype(np.float))
    Xval = rescaler.transform(Xval.astype(np.float))
    Xtest = rescaler.transform(Xtest.astype(np.float))

    # save for future use on new data
    dump(rescaler, './processed_data/std_scaler.bin', compress=True)

    return Xtrain, Xval, Xtest


def plot_accuracy_scores(y_test: np.array, y_pred: np.array,
                         cmap: str) -> None:
    """ Prints accuracy scores and plots a confusion matrix

    Arguments:
        y_test {np.array} -- [description]
        y_pred {np.array} -- [description]
        cmap {str} -- [description]

    Returns:
        [None] -- [description]
    """
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred, average='macro'))
    # precision_score, recall_score, f1_score
    plt.figure(figsize=(9, 6))
    heatmap(
        confusion_matrix(y_test, y_pred),
        # annot=True,
        cmap=cmap,
        # fmt='g',
    )

    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')

    return f1_score(y_test, y_pred, average='macro')


def print_metric_scores(
        y_test: np.array,
        y_pred: np.array,
        average: str = 'macro') -> Tuple[float, float, float, float]:
    """[summary]

    Arguments:
        y_test {np.array} -- [description]
        y_pred {np.array} -- [description]

    Keyword Arguments:
        average {str} -- [description] (default: {'macro'})

    Returns:
        Tuple[float, float, float, float] -- [description]
    """

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average=average, labels=np.unique(y_test))
    recall = recall_score(y_test, y_pred, average=average,
                          labels=np.unique(y_test))
    f1 = f1_score(y_test, y_pred, average=average, labels=np.unique(y_test))

    print('Accuracy: {}'.format(np.round(acc, 4)))
    print('Precision: {}'.format(np.round(precision, 4)))
    print('Recall: {}'.format(np.round(recall, 4)))
    print('F1 score: {}'.format(np.round(f1, 4)))

    return acc, precision, recall, f1


def plot_training_history(
        histories: List,
        labels: List,
        plot_title: str = 'Training History',
        show_val: bool = False) -> None:
    """[summary]

    Arguments:
        histories {List} -- [description]
        labels {List} -- [description]

    Keyword Arguments:
        plot_title {str} -- [description] (default: {'Training History'})
        show_val {bool} -- [description] (default: {False})
    """
    plt.figure(figsize=(14, 6))

    for i, history in enumerate(histories):
        plt.subplot(1, 2, 1)
        plt.plot(history['acc'], label=labels[i])
        if show_val:
            plt.plot(history['val_acc'], label="val accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label=labels[i])
        if show_val:
            plt.plot(history['val_loss'], label="val loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()

    plt.suptitle(plot_title)


def dataframe_from_scores(a: np.array,
                          p: np.array,
                          r: np.array,
                          f1: np.array,
                          y_t: np.array) -> pd.DataFrame:
    """Takes calculated scores from a model and creates a pandas dataframe,
    which contains a full class by class breakdown with a human readable label
    for each row.

    Arguments:
        a {np.array} -- [description]
        p {np.array} -- [description]
        r {np.array} -- [description]
        f1 {np.array} -- [description]
        y_t {np.array} -- [description]

    Returns:
        pd.DataFrame -- [description]
    """
    from os import path
    # TODO: allow user to put path to class dict, rather than
    # being hardcoded
    class_dict = np.load(
        path.join('./processed_data', 'class_dict.npy'),
        allow_pickle=True).item()
    species_scores: Dict = {}
    for i, species in enumerate(np.unique(y_t)):
        species_scores[class_dict[species]] = {
            'recall': r[i],
            'precision': p[i],
            'f1': f1[i]}

    df: pd.DataFrame = pd.DataFrame(species_scores)
    df = df.transpose()

    return df
