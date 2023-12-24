import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class ImagePreprocessor:
    def __init__(self):
        pass

    def preprocess(self, image_path):
        """
        Preprocess a single image from the given path.

        Operations include adaptive thresholding, morphological transformations,
        dilation, and Gaussian blurring.

        :param image_path: Path to the image file
        :return: Preprocessed image ready for segmentation
        """
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        # Adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

        # Morphological closing
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # Dilation
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)

        # Gaussian blurring
        image = cv2.GaussianBlur(image, (5, 5), 0)

        return image

    def segment_characters(self, image):
        """
        Segment the preprocessed image into individual characters.

        :param image: Preprocessed image
        :return: List of segmented character images
        """
        # Define character segments based on heuristic locations
        segments = [
            image[10:50, 30:50],
            image[10:50, 50:70],
            image[10:50, 70:90],
            image[10:50, 90:110],
            image[10:50, 110:130],
        ]

        # Convert segments to array format
        return [tf.keras.preprocessing.image.img_to_array(Image.fromarray(segment)) for segment in segments]
