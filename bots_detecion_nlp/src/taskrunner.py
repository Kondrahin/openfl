"""Copyright (C) 2020-2021 Intel Corporation
   SPDX-License-Identifier: Apache-2.0

Licensed subject to the terms of the separately executed evaluation
license agreement between Intel Corporation and you.
"""
from tensorflow import keras

from openfl.federated import KerasTaskRunner

def f1(y_true, y_pred):
    # Ensure y_true is float32
    y_true = keras.backend.cast(y_true, 'float32')

    def recall(y_true, y_pred):
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + keras.backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.backend.epsilon())
        return precision

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + keras.backend.epsilon()))


def build_model(shape):
    """
    Define the model architecture.

    Args:
        input_shape (numpy.ndarray): The shape of the data
        num_classes (int): The number of classes of the dataset
    Returns:
        tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras
    """

    model = keras.models.Sequential([
        keras.layers.Dense(128, input_shape=(shape,), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Компилируем модель
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC', f1])

    return model


class Keras(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, latent_dim, **kwargs):
        """
        Init taskrunner.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = build_model(self.data_loader.get_labels_shape())

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=self.logger.info)

