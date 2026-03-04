import argparse
from pathlib import Path

from configs import FontProcessingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.image import GlyphImageGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image dataset for training")
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument(
        "--reference_fonts_dir", type=str, help="Reference fonts directory"
    )
    parser.add_argument("--source_charset_path", type=str, help="Source charset path")
    parser.add_argument(
        "--img_size", type=int, nargs=2, help="Image size (width height)"
    )
    parser.add_argument("--sample_ratio", type=float, help="Sampling ratio (0-1)")
    return parser.parse_args()


def prepare_image_dataset(
    target_font_path: str,
    reference_fonts_dir: str,
    source_charset_path: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    tgt_generator = GlyphImageGenerator.from_target_font(
        target_font_path=target_font_path,
        font_processing_config=font_processing_config,
    )
    ref_generators = GlyphImageGenerator.from_reference_fonts(
        reference_fonts_dir=reference_fonts_dir,
        font_processing_config=font_processing_config,
    )

    # 计算目标字体和参考字体的字符集交集
    print("[FontGen] Calculating charset intersection...")
    tgt_charset = tgt_generator.get_supported_charset()
    print(f"[FontGen] Target font supports {len(tgt_charset)} characters")
    
    # 计算所有参考字体的字符集并集
    ref_charset_union = set()
    for ref_generator in ref_generators:
        ref_charset = ref_generator.get_supported_charset()
        print(f"[FontGen] Reference font {ref_generator.font_name} supports {len(ref_charset)} characters")
        ref_charset_union.update(ref_charset)
    print(f"[FontGen] Combined reference fonts support {len(ref_charset_union)} characters")
    
    # 计算交集
    common_charset = tgt_charset.intersection(ref_charset_union)
    print(f"[FontGen] Common charset size: {len(common_charset)} characters")
    
    # 保存交集到临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
        for char in common_charset:
            f.write(char + '\n')
        temp_charset_path = f.name
    
    try:
        # 使用交集字符集生成目标字体图像
        print(f"[FontGen] Generating target font images...")
        tgt_generator.generate_glyph_images(
            source_charset_path=temp_charset_path,
            font_role="target",
        )
        
        # 获取目标字体实际生成的字符（过滤掉无法渲染的字符）
        # 我们需要先让目标字体生成图像，然后收集实际生成的字符
        target_output_dir = Path(font_processing_config.data_root) / "target"
        generated_target_chars = set()
        for img_file in target_output_dir.glob("*.png"):
            # 从文件名中提取字符编码
            char_code = int(img_file.stem, 16)
            generated_target_chars.add(chr(char_code))
        print(f"[FontGen] Target font actually generated {len(generated_target_chars)} characters")
        
        # 对于每个参考字体，只使用目标字体实际生成的字符
        for ref_generator in ref_generators:
            print(f"[FontGen] Generating reference font {ref_generator.font_name} images...")
            # 获取参考字体实际支持的字符
            ref_supported = ref_generator.get_supported_charset()
            # 过滤出目标字体实际生成的字符中参考字体也支持的部分
            ref_common = generated_target_chars.intersection(ref_supported)
            print(f"[FontGen] Reference font {ref_generator.font_name} will generate {len(ref_common)} characters (matching target font)")
            
            # 保存过滤后的字符集到临时文件
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
                for char in ref_common:
                    f.write(char + '\n')
                ref_temp_path = f.name
            
            try:
                ref_generator.generate_glyph_images(
                    source_charset_path=ref_temp_path,
                    font_role="reference",
                )
            finally:
                import os
                os.unlink(ref_temp_path)
    finally:
        # 清理临时文件
        import os
        os.unlink(temp_charset_path)


def main() -> None:
    args = parse_args()
    font_processing_config = update_config_from_args(
        converting_config=FontProcessingConfig(),
        args=args,
    )

    prepare_image_dataset(
        target_font_path=args.target_font_path,
        reference_fonts_dir=args.reference_fonts_dir,
        source_charset_path=args.source_charset_path,
        font_processing_config=font_processing_config,
    )


if __name__ == "__main__":
    main()
