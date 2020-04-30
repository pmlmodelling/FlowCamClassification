from os import path
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Tuple


class KerasModel():
    """ A wrapper around the Keras functioanlity to compile and train
    TensorFlow deep learning models """

    def __init__(self, model, verbose=True):
        self.model = model

        if verbose:
            model.summary()

    def compile_model(self, compile_opt: Dict = {}) -> None:
        """Compiles the TF, Keras model based on the options provided

        Keyword Arguments:
            compile_opt {Dict} -- [description] (default: {{}})
        """
        # if no arguments passed, some defaults are provided.
        OPT = compile_opt.get('OPT', 'adam')
        loss = compile_opt.get('loss', 'categorical_crossentropy')
        metrics = compile_opt.get('metrics', ['accuracy'])

        self.model.compile(optimizer=OPT, loss=loss,  metrics=metrics)

    def evaluate_model(self,
                       X_test: np.array,
                       y_test: np.array,
                       model: Model,
                       eval_opt: Dict = {}) -> float:
        """Method for evaluating the CNN trained model

        Arguments:
            X_test {np.array} -- [description]
            y_test {np.array} -- [description]
            model {Model} -- [description]

        Keyword Arguments:
            eval_opt {Dict} -- [description] (default: {{}})

        Returns:
            float -- [description]
        """
        BATCH_SIZE = eval_opt.get('BATCH_SIZE', 128)
        VERBOSE = eval_opt.get('VERBOSE', 1)

        score = self.model.evaluate(
            X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
        return score

    def load_model(self, model_filepath: path) -> None:
        """[summary]

        Arguments:
            model_filepath {path} -- [description]
        """
        model_path = path.join(
            './trained_models', '{}_model.h5'.format(model_filepath))
        self.model = load_model(model_path)

    def save_model(self, model_filepath: path) -> None:
        """[summary]

        Arguments:
            model_filepath {path} -- [description]
        """
        self.model.save(model_filepath)

    def train_model(self,
                    training_generator: Tuple[np.array, np.array],
                    validation_data: np.array,
                    model_name: str,
                    y_train: np.array,
                    save_model: bool = True,
                    training_opt: Dict = {}) -> List:
        """[summary]

        Arguments:
            training_generator {Tuple[np.array, np.array]} -- [description]
            validation_data {np.array} -- [description]
            model_name {str} -- [description]
            y_train {np.array} -- [description]

        Keyword Arguments:
            save_model {bool} -- [description] (default: {True})
            training_opt {Dict} -- [description] (default: {{}})

        Returns:
            List -- [description]
        """
        # sets default values if training options were not passed
        BATCH_SIZE = training_opt.get('BATCH_SIZE', 16)
        NB_EPOCH = training_opt.get('NB_EPOCH', 250)
        VERBOSE = training_opt.get('VERBOSE', 1)

        output_path = path.join('./trained_models', '{}'.format(model_name))

        y = [np.where(r == 1)[0][0] for r in y_train]
        for v in np.where(~y_train.any(axis=0))[0]:
            for _ in range(1000):
                y.append(v)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y),
                                                          y)
        class_weights = dict(enumerate(class_weights))

        # setup model training callbacks
        # ------------------------------
        # save the best model so far when training
        checkpoint = ModelCheckpoint(
            output_path, monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max')

        # lower learning rate when models learning has plateaued
        lr_drop = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=8, min_lr=0.000001)

        # stop training if signs of model convergence
        early_stopping = EarlyStopping(monitor='loss', patience=13)

        # enables tensorboard from console for diagnostic tools
        tensor_board = TensorBoard(log_dir='Graph', histogram_freq=0,
                                   write_graph=True, write_images=True)

        callbacks_list = [checkpoint, lr_drop, early_stopping, tensor_board]

        # TODO: fit_generator will become deprecated, model.fit now supports
        # generators, so change this over.
        history = self.model.fit_generator(
            training_generator,
            steps_per_epoch=y_train.shape[0] // BATCH_SIZE,
            validation_data=validation_data,
            validation_steps=32,
            epochs=NB_EPOCH,
            verbose=VERBOSE,
            callbacks=callbacks_list,
            class_weight=class_weights)

        return history

    def train_model_with_no_augmentation(self,
                                         X_train: np.array,
                                         y_train: np.array,
                                         X_val: np.array,
                                         y_val: np.array,
                                         model_name: str,
                                         training_opt: Dict = {}) -> List:
        """Same as above but with no image augmentation, not usually used
        TODO: make above method flexible so it can handle this

        Arguments:
            X_train {np.array} -- [description]
            y_train {np.array} -- [description]
            X_val {np.array} -- [description]
            y_val {np.array} -- [description]
            model_name {str} -- [description]

        Keyword Arguments:
            training_opt {Dict} -- [description] (default: {{}})

        Returns:
            List -- [description]
        """
        BATCH_SIZE = training_opt.get('BATCH_SIZE', 46)
        NB_EPOCH = training_opt.get('NB_EPOCH', 250)
        VERBOSE = training_opt.get('VERBOSE', 1)

        output_path = path.join(
            './trained_models', '{}_model.h5'.format(model_name))

        checkpoint = ModelCheckpoint(
            output_path, monitor='val_acc', verbose=1,
            save_best_only=True, mode='max')
        y = [np.where(r == 1)[0][0] for r in y_train]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y),
                                                          y)
        callbacks_list = [checkpoint]

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            epochs=NB_EPOCH,
            verbose=VERBOSE,
            callbacks=callbacks_list,
            class_weight=class_weights)
        return history
