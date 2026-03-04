#!/usr/bin/env python3
import argparse
from pathlib import Path
from fontTools.ttLib import TTFont

def extract_charset_from_font(font_path):
    """从字体文件中提取字符集"""
    font = TTFont(font_path)
    chars = []
    
    # 使用getBestCmap()获取最佳字符映射
    best_cmap = font.getBestCmap()
    
    # 遍历所有字符
    for code, name in best_cmap.items():
        # 只添加有效的Unicode字符
        if 0x20 <= code <= 0xFFFF:
            chars.append(chr(code))
    
    return chars

def save_charset(chars, output_path):
    """保存字符集到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in chars:
            f.write(char + '\n')

def main():
    parser = argparse.ArgumentParser(description="从字体文件中提取字符集")
    parser.add_argument('--font_path', type=str, required=True, help="字体文件路径")
    parser.add_argument('--output_path', type=str, required=True, help="输出字符集文件路径")
    args = parser.parse_args()
    
    font_path = Path(args.font_path)
    output_path = Path(args.output_path)
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 提取字符集
    chars = extract_charset_from_font(font_path)
    
    # 保存字符集
    save_charset(chars, output_path)
    
    print(f"从 {font_path} 中提取了 {len(chars)} 个字符")
    print(f"字符集已保存到 {output_path}")

if __name__ == "__main__":
    main()
