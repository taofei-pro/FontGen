import argparse

from configs.vqgan2_config import VQGAN2ModelConfig, VQGAN2TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQGAN-2")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    model_config = VQGAN2ModelConfig()
    training_config = VQGAN2TrainingConfig()
    print("VQGAN-2 training scaffold initialized")
    print(model_config)
    print(training_config)


if __name__ == "__main__":
    main()
