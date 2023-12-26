from captcha_recognision.captcha_solver_cnn import CNNCaptchaSolver
from captcha_recognision.captcha_solver_knn import KNNCaptchaSolver
from captcha_recognision.data_manager.dataset_manager import DatasetManager
from captcha_recognision.config import config


def main():
    dataset_manager = DatasetManager(
        dataset_path=config["dataset_path"],
        dataset_name=None,
    )

    cnn_solver = CNNCaptchaSolver(dataset_manager)
    cnn_solver.train_model()
    cnn_solver.test_random_sample()
    cnn_solver.evaluate_on_test_dataset()

    knn_solver = KNNCaptchaSolver(dataset_manager)
    knn_solver.train()
    knn_solver.test_random_sample()
    knn_solver.evaluate_on_test_dataset()


if __name__ == "__main__":
    main()
