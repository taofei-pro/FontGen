import argparse

from configs.converting_config import ConvertingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.image import GlyphImageConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert images to SVG format")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--blacklevel", type=float, default=None)
    parser.add_argument("--turdsize", type=int, default=None)
    parser.add_argument("--alphamax", type=float, default=None)
    parser.add_argument("--opttolerance", type=float, default=None)
    return parser.parse_args()


def convert_to_svg(
    input_dir: str,
    output_dir: str,
    converting_config: ConvertingConfig,
) -> None:
    converter = GlyphImageConverter(converting_config=converting_config)
    converter.convert_images_to_svgs(input_dir=input_dir, output_dir=output_dir)


def main() -> None:
    args = parse_args()
    converting_config = update_config_from_args(
        converting_config=ConvertingConfig(),
        args=args,
    )
    convert_to_svg(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        converting_config=converting_config,
    )


if __name__ == "__main__":
    main()
