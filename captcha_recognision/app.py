from captcha_solver import CaptchaSolver

def main():
    dataset_path = "path/to/dataset"
    model_path = "path/to/saved/model"

    solver = CaptchaSolver(dataset_path, model_path)
    solver.train_model()

    captcha_image = "path/to/captcha/image"
    solved_text = solver.solve_captcha(captcha_image)
    print(f"Solved CAPTCHA: {solved_text}")

if __name__ == "__main__":
    main()
