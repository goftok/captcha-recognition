from captcha_recognision.captcha_solver import CaptchaSolver
from captcha_recognision.config import config


def main():
    solver = CaptchaSolver()
    solver.train_model()

    _, _, test_dataset = solver.dataset_manager.split_dataset(
        train_split=config["train_split"],
        val_split=config["val_split"],
        test_split=config["test_split"],
    )

    solver.test_random_sample(test_dataset)
    solver.evaluate_on_test_dataset(test_dataset)


if __name__ == "__main__":
    main()
