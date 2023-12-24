from captcha_recognision.captcha_solver import CaptchaSolver
from captcha_recognision.preprocessing.image_preprocessor import ImagePreprocessor
from captcha_recognision.model.captcha_model import CaptchaModel
from captcha_recognision.data_manager.dataset_manager import DatasetManager
from captcha_recognision.config import config
from captcha_recognision.utils.console import console
import numpy as np
import random

from sklearn.metrics import classification_report


def main():
    dataset_manager = DatasetManager(
        dataset_path=config["dataset_path"],
        dataset_name=None,
    )

    train_dataset, val_dataset, test_dataset = dataset_manager.split_dataset(
        train_split=config["train_split"],
        val_split=config["val_split"],
        test_split=config["test_split"],
    )

    captcha_model = CaptchaModel(
        model_path=config["model_path"],
        input_shape=dataset_manager.get_input_shape(),
        num_classes=dataset_manager.get_num_classes(),
    )

    if not config["model_path"]:
        captcha_model.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
        )

        # Save model
        captcha_model.save()

    # Test on a random sample from the training dataset
    index = random.randint(0, len(train_dataset[0]) - 1)
    image = train_dataset[0][index]

    image_preprocessed = np.expand_dims(image, axis=0)

    predicted_text = captcha_model.predict(image_preprocessed, dataset_manager.info)
    decoded_label = dataset_manager.decode_label(train_dataset[1][index])

    console.print(f"Predicted text: {predicted_text}")
    console.print(f"Actual text:    {decoded_label}")

    # Test on the test dataset
    # List to store predictions
    test_predictions = []

    # Iterate over each image in the test dataset
    for image, label in zip(test_dataset[0], test_dataset[1]):
        # Preprocess the image and add a batch dimension, if necessary
        image_preprocessed = np.expand_dims(image, axis=0)

        # Predict
        predicted_text = captcha_model.predict(image_preprocessed, dataset_manager.info, verbose=0)

        # Decode the label
        decoded_label = dataset_manager.decode_label(label)

        # append to comapare later
        test_predictions.append((predicted_text, decoded_label))

    print(classification_report([x[0] for x in test_predictions], [x[1] for x in test_predictions]))


if __name__ == "__main__":
    main()
