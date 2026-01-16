import argparse

from models.vectorize.fontforge_pipeline import svg_to_font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build font from SVG directory")
    parser.add_argument("--svg_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svg_to_font(svg_dir=args.svg_dir, output_path=args.output_path)


if __name__ == "__main__":
    main()
