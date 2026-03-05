import argparse
from pathlib import Path
from tqdm import tqdm

from utils.image.image_generator import GlyphImageGenerator
from configs.font_processing_config import FontProcessingConfig
from utils.font.font_utils import read_charset_from_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--charset_path', type=str, default='charsets/target_charset.txt', help='Charset file path')
    parser.add_argument('--reference_fonts_dir', type=str, default='fonts/reference', help='Reference fonts directory')
    parser.add_argument('--output_dir', type=str, default='data/reference_infer', help='Output directory for reference images')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载字符集
    charset = read_charset_from_file(args.charset_path)
    
    # 配置字体处理
    font_processing_config = FontProcessingConfig()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成reference图片
    ref_generators = GlyphImageGenerator.from_reference_fonts(
        reference_fonts_dir=args.reference_fonts_dir,
        font_processing_config=font_processing_config,
    )
    
    # 为每个字符生成reference图片
    for char in tqdm(charset, desc='Generating reference images'):
        for ref_generator in ref_generators:
            # 检查字符是否在字体的覆盖范围内
            covered_charset_path = (
                Path(font_processing_config.unihan_coverage_charset_dir)
                / ref_generator.font_name
                / 'covered.txt'
            )
            if covered_charset_path.exists():
                covered_charset = read_charset_from_file(covered_charset_path)
                if char in covered_charset:
                    ref_generator.save_glyph_image(
                        char=char,
                        output_dir=output_dir,
                        img_size=font_processing_config.img_size,
                    )
                    break  # 只使用第一个能覆盖该字符的字体
            else:
                # 如果覆盖文件不存在，尝试直接生成图像
                try:
                    ref_generator.save_glyph_image(
                        char=char,
                        output_dir=output_dir,
                        img_size=font_processing_config.img_size,
                    )
                    break  # 只使用第一个能生成该字符的字体
                except Exception as e:
                    print(f"Error generating image for character '{char}': {e}")
                    continue

if __name__ == '__main__':
    main()