import argparse

from configs.sr_config import SRModelConfig, SRTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR model")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    model_config = SRModelConfig()
    training_config = SRTrainingConfig()
    print("SR training scaffold initialized")
    print(model_config)
    print(training_config)


if __name__ == "__main__":
    main()
