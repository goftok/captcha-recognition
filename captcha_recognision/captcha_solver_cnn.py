import random
import numpy as np
from sklearn.metrics import classification_report

from captcha_recognision.config import config
from captcha_recognision.utils.console import console
from captcha_recognision.model.captcha_model import CaptchaModel
from captcha_recognision.data_manager.dataset_manager import DatasetManager


class CNNCaptchaSolver:
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.captcha_model = CaptchaModel(
            model_path=config["model_path"],
            input_shape=self.dataset_manager.get_input_shape(),
            num_classes=self.dataset_manager.get_num_classes(),
        )

        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.split_dataset(
            train_split=config["train_split"],
            val_split=config["val_split"],
            test_split=config["test_split"],
        )

    def train_model(self):
        if not config["model_path"]:
            self.captcha_model.fit(
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
            )

            # Save model
            self.captcha_model.save()

    def test_random_sample(self):
        index = random.randint(0, len(self.train_dataset[0]) - 1)
        image = self.train_dataset[0][index]
        image_preprocessed = np.expand_dims(image, axis=0)

        label_encoding_map = self.dataset_manager.get_label_encoding_map()
        predicted_text = self.captcha_model.predict(image_preprocessed, label_encoding_map)
        decoded_label = self.dataset_manager.decode_label(self.train_dataset[1][index])

        console.print(f"Predicted text: {predicted_text}")
        console.print(f"Actual text:    {decoded_label}")

    def evaluate_on_test_dataset(self):
        label_encoding_map = self.dataset_manager.get_label_encoding_map()
        test_predictions = []

        for image, label in zip(self.test_dataset[0], self.test_dataset[1]):
            image_preprocessed = np.expand_dims(image, axis=0)
            predicted_text = self.captcha_model.predict(image_preprocessed, label_encoding_map, verbose=0)
            decoded_label = self.dataset_manager.decode_label(label)
            test_predictions.append((predicted_text, decoded_label))

        print(classification_report([x[0] for x in test_predictions], [x[1] for x in test_predictions]))
