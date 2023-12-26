import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from captcha_recognision.config import config
from captcha_recognision.utils.console import console
from captcha_recognision.data_manager.dataset_manager import DatasetManager


class KNNCaptchaSolver:
    def __init__(self, dataset_manager: DatasetManager, n_neighbors=5):
        self.dataset_manager = dataset_manager
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.train_dataset, self.test_dataset = self._prepare_datasets()

    def _prepare_datasets(self):
        """
        Prepare the train and test datasets for KNN classification.

        :return: Tuple of train and test datasets
        """

        train_dataset, _, test_dataset = self.dataset_manager.split_dataset(
            train_split=0.8,
            val_split=config["val_split"],
            test_split=0.2,
        )

        # Flatten the datasets
        train_dataset = (train_dataset[0].reshape(train_dataset[0].shape[0], -1), train_dataset[1])
        test_dataset = (test_dataset[0].reshape(test_dataset[0].shape[0], -1), test_dataset[1])

        return train_dataset, test_dataset

    def train(self):
        """
        Train the KNN model.
        """
        try:
            self.knn.fit(*self.train_dataset)
        except Exception as e:
            console.print(f"Error during training: {e}")

    def predict(self, X):
        """
        Predict the text in the given image.

        :param X: Image to predict
        :return: Predicted text
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        try:
            predictions = self.knn.predict(X)
            return "".join(self.dataset_manager.decode_label(pred) for pred in predictions)
        except Exception as e:
            console.print(f"Error during prediction: {e}")
            return ""

    def test_random_sample(self):
        index = random.randint(0, len(self.train_dataset[0]) - 1)
        image = self.train_dataset[0][index]
        predicted_text = self.predict(image)
        decoded_label = self.dataset_manager.decode_label(self.train_dataset[1][index])

        console.print(f"Predicted text: {predicted_text}")
        console.print(f"Actual text:    {decoded_label}")

    def evaluate_on_test_dataset(self):
        test_predictions = []

        for image, label in zip(*self.test_dataset):
            predicted_text = self.predict(image)
            decoded_label = self.dataset_manager.decode_label(label)
            test_predictions.append((predicted_text, decoded_label))

        print(classification_report([x[0] for x in test_predictions], [x[1] for x in test_predictions]))
