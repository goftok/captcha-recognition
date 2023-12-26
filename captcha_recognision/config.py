import os

# get dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    # "dataset_name": "hammer888/captcha-data",
    "dataset_path": f"{BASE_DIR}/samples",
    "model_path": "/Users/goftok/github/captcha-recognition/captcha_model.keras",
    # "model_path": None,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "epochs": 50,
    "batch_size": 128,
}
