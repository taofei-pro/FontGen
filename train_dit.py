import argparse

from configs.dit_config import DiTModelConfig, DiTTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiT")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    model_config = DiTModelConfig()
    training_config = DiTTrainingConfig()
    print("DiT training scaffold initialized")
    print(model_config)
    print(training_config)


if __name__ == "__main__":
    main()
