import tensorflow as tf
import numpy as np
from captcha_recognision.utils.console import console


class CaptchaModel:
    def __init__(self, input_shape: tuple, num_classes: int, model_path=None):
        self.model = self._load_model(model_path) if model_path else self._create_model(input_shape, num_classes)

    def _create_model(self, input_shape: tuple, num_classes: int = 36) -> tf.keras.Sequential:
        """
        Create a CNN model for CAPTCHA recognition.

        :param input_shape: Tuple specifying the input shape (height, width, channels)
        :param num_classes: Number of distinct characters in the CAPTCHA
        :return: CNN model
        """
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", input_shape=input_shape))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(1500))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(num_classes))
        model.add(tf.keras.layers.Activation("softmax"))

        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )

        console.print(model.summary())

        # model = Sequential()

        # # Convolutional layer 1
        # model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # # Convolutional layer 2
        # model.add(Conv2D(64, (3, 3), activation="relu"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D((2, 2)))

        # # Convolutional layer 3
        # model.add(Conv2D(128, (3, 3), activation="relu"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D((2, 2)))

        # # Flattening and Dense layers
        # model.add(Flatten())
        # model.add(Dense(64, activation="relu"))
        # model.add(Dropout(0.5))
        # model.add(Dense(num_classes, activation="softmax"))

        # # Compile the model
        # model.compile(
        #     optimizer="adam",
        #     loss="categorical_crossentropy",
        #     metrics=["accuracy"],
        # )

        # console.print(model.summary())

        return model

    def _load_model(self, model_path) -> tf.keras.Sequential:
        """
        Load a pre-trained model from the given path.

        :param model_path: Path to the model file
        :return: Loaded model
        """
        return tf.keras.models.load_model(model_path)

    def fit(self, train_dataset, val_dataset, epochs=10, batch_size=32) -> None:
        """
        Train the model on the given dataset.

        :param training_data: Training data
        :param validation_data: Validation data
        :param epochs: Number of epochs to train
        :param batch_size: Batch size for training
        """
        assert self.model is not None, "Model not loaded"

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        self.model.fit(
            train_dataset[0],
            train_dataset[1],
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
        )

    def predict(self, image, info, verbose=1):
        """
        Predict the CAPTCHA text from a given image.

        :param image: The preprocessed image (or batch of images) of the CAPTCHA.
        :param info: Dictionary mapping label encodings back to the original characters.
        :return: Predicted text of the CAPTCHA.
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image, verbose=verbose)
        predicted_text = "".join(info[np.argmax(pred)] for pred in predictions)

        return predicted_text

    def save(self, model_path=None):
        """
        Save the model to the given path.

        :param model_path: Path to save the model
        """
        assert self.model is not None, "Model not loaded"

        if model_path:
            self.model.save(model_path)
        else:
            self.model.save("captcha_model.keras")
