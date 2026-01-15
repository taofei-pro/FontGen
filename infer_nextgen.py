import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-gen inference pipeline")
    parser.add_argument("--charset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs_nextgen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Next-gen inference scaffold initialized")
    print(args)


if __name__ == "__main__":
    main()
