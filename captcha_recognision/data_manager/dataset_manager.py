import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from captcha_recognision.preprocessing.image_preprocessor import ImagePreprocessor


class DatasetManager:
    def __init__(self, dataset_name: str = None, dataset_path: str = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.info = None
        self.input_shape = None
        self.categories = None
        self.images = None
        self.labels = None
        self.image_preprocessor = ImagePreprocessor()
        self.load_dataset()

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from a local path or Hugging Face's datasets library.
        """
        if self.dataset_path:
            self._load_local_dataset()
        else:
            load_dataset(self.dataset_name)

    def _load_local_dataset(self):
        """
        Load dataset from a local directory where file names are labels.
        Supports multiple image file extensions.
        """
        X, y = [], []  # Initialize X and y as empty lists
        path = Path(self.dataset_path)
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

        for ext in valid_extensions:
            for img_path in path.glob(f"*{ext}"):
                try:
                    img_path_str = str(img_path)
                    preprocessed_image = self.image_preprocessor.preprocess(img_path_str)
                    characters = self.image_preprocessor.segment_characters(preprocessed_image)
                    label = img_path.stem  # Extracts file name without extension
                    for char_image, char_label in zip(characters, label):
                        X.append(char_image)
                        y.append(char_label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        # Convert X and y to NumPy arrays and process
        X = np.array(X)
        y = np.array(y)
        X = X.astype("float32")
        X /= 255

        y_le = LabelEncoder().fit_transform(y)
        y_ohe = OneHotEncoder(sparse=False).fit_transform(y_le.reshape(len(y_le), 1))

        self.input_shape = (X.shape[1], X.shape[2], 1)
        self.categories = y_ohe.shape[1]
        self.info = {y_le[i]: y[i] for i in range(len(y))}
        self.images = X
        self.labels = y_ohe

    def split_dataset(self, train_split=0.7, val_split=0.15, test_split=0.15):
        """
        Split the dataset into training, validation, and test sets.
        """
        # Ensure that the dataset is loaded and in the correct format
        # assert self.images , "Dataset not loaded"
        # assert isinstance(self.dataset, pd.DataFrame), "Dataset is not a DataFrame"

        # Ensure that the splits add up to 1
        assert train_split + val_split + test_split == 1, "Splits must add up to 1"

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.labels, test_size=(val_split + test_split), random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=test_split / (val_split + test_split), random_state=42
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_input_shape(self):
        """
        Get the input shape of the dataset.
        """
        assert self.input_shape is not None, "Input shape not defined"
        return self.input_shape

    def get_num_classes(self):
        """
        Get the number of classes in the dataset.
        """
        assert self.categories is not None, "Number of classes not defined"
        return self.categories

    def get_info(self):
        """
        Get the info of the dataset.
        """
        assert self.info is not None, "Info not defined"
        return self.info

    def decode_label(self, encoded_label):
        # Assuming encoded_label is one-hot encoded
        label_encoded = np.argmax(encoded_label)
        original_label = self.info[label_encoded]
        return original_label
