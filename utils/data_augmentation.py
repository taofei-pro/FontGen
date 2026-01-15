"""
数据增强模块
提供旋转、缩放、噪声、亮度调整等增强技术
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional


class DataAugmentation:
    """数据增强类"""
    
    def __init__(self, 
                 rotation_range: float = 5.0,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 noise_std: float = 0.01,
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 apply_prob: float = 0.5):
        """
        初始化数据增强参数
        
        Args:
            rotation_range: 旋转角度范围（度）
            scale_range: 缩放范围
            noise_std: 噪声标准差
            brightness_range: 亮度调整范围
            apply_prob: 应用增强的概率
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.apply_prob = apply_prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对图像应用数据增强
        
        Args:
            image: 输入图像张量 [C, H, W]
            
        Returns:
            增强后的图像张量
        """
        if random.random() > self.apply_prob:
            return image
        
        # 随机旋转
        if random.random() < 0.3:
            image = self._rotate_image(image)
        
        # 随机缩放
        if random.random() < 0.3:
            image = self._scale_image(image)
        
        # 随机噪声
        if random.random() < 0.3:
            image = self._add_noise(image)
        
        # 随机亮度调整
        if random.random() < 0.3:
            image = self._adjust_brightness(image)
        
        return image
    
    def _rotate_image(self, image: torch.Tensor) -> torch.Tensor:
        """旋转图像"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # 计算旋转中心
        center = (image.shape[2] // 2, image.shape[1] // 2)
        
        # 创建旋转矩阵
        theta = torch.tensor([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
        ], dtype=torch.float32)
        
        # 应用旋转
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        rotated = F.grid_sample(image.unsqueeze(0), grid, align_corners=False)
        
        return rotated.squeeze(0)
    
    def _scale_image(self, image: torch.Tensor) -> torch.Tensor:
        """缩放图像"""
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 计算缩放后的尺寸
        h, w = image.shape[1], image.shape[2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放图像
        scaled = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), 
                             mode='bilinear', align_corners=False)
        
        # 如果缩放后尺寸不同，进行裁剪或填充
        if new_h != h or new_w != w:
            if new_h > h or new_w > w:
                # 裁剪到原始尺寸
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                scaled = scaled[:, :, start_h:start_h+h, start_w:start_w+w]
            else:
                # 填充到原始尺寸
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                scaled = F.pad(scaled, (pad_w, w-new_w-pad_w, pad_h, h-new_h-pad_h))
        
        return scaled.squeeze(0)
    
    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = image + noise
        
        # 确保像素值在[0, 1]范围内
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        return noisy_image
    
    def _adjust_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """调整亮度"""
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # 调整亮度
        brightened = image * brightness_factor
        
        # 确保像素值在[0, 1]范围内
        brightened = torch.clamp(brightened, 0, 1)
        
        return brightened


class AdvancedDataAugmentation:
    """高级数据增强类"""
    
    def __init__(self, 
                 rotation_range: float = 5.0,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 noise_std: float = 0.01,
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 apply_prob: float = 0.7):
        """
        初始化高级数据增强参数
        
        Args:
            rotation_range: 旋转角度范围（度）
            scale_range: 缩放范围
            noise_std: 噪声标准差
            brightness_range: 亮度调整范围
            contrast_range: 对比度调整范围
            apply_prob: 应用增强的概率
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.apply_prob = apply_prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对图像应用高级数据增强
        
        Args:
            image: 输入图像张量 [C, H, W]
            
        Returns:
            增强后的图像张量
        """
        if random.random() > self.apply_prob:
            return image
        
        # 随机旋转
        if random.random() < 0.4:
            image = self._rotate_image(image)
        
        # 随机缩放
        if random.random() < 0.4:
            image = self._scale_image(image)
        
        # 随机噪声
        if random.random() < 0.4:
            image = self._add_noise(image)
        
        # 随机亮度调整
        if random.random() < 0.4:
            image = self._adjust_brightness(image)
        
        # 随机对比度调整
        if random.random() < 0.4:
            image = self._adjust_contrast(image)
        
        return image
    
    def _rotate_image(self, image: torch.Tensor) -> torch.Tensor:
        """旋转图像"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # 计算旋转中心
        center = (image.shape[2] // 2, image.shape[1] // 2)
        
        # 创建旋转矩阵
        theta = torch.tensor([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
        ], dtype=torch.float32)
        
        # 应用旋转
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        rotated = F.grid_sample(image.unsqueeze(0), grid, align_corners=False)
        
        return rotated.squeeze(0)
    
    def _scale_image(self, image: torch.Tensor) -> torch.Tensor:
        """缩放图像"""
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 计算缩放后的尺寸
        h, w = image.shape[1], image.shape[2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放图像
        scaled = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), 
                             mode='bilinear', align_corners=False)
        
        # 如果缩放后尺寸不同，进行裁剪或填充
        if new_h != h or new_w != w:
            if new_h > h or new_w > w:
                # 裁剪到原始尺寸
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                scaled = scaled[:, :, start_h:start_h+h, start_w:start_w+w]
            else:
                # 填充到原始尺寸
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                scaled = F.pad(scaled, (pad_w, w-new_w-pad_w, pad_h, h-new_h-pad_h))
        
        return scaled.squeeze(0)
    
    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = image + noise
        
        # 确保像素值在[0, 1]范围内
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        return noisy_image
    
    def _adjust_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """调整亮度"""
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # 调整亮度
        brightened = image * brightness_factor
        
        # 确保像素值在[0, 1]范围内
        brightened = torch.clamp(brightened, 0, 1)
        
        return brightened
    
    def _adjust_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """调整对比度"""
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # 计算图像均值
        mean = image.mean()
        
        # 调整对比度
        contrasted = (image - mean) * contrast_factor + mean
        
        # 确保像素值在[0, 1]范围内
        contrasted = torch.clamp(contrasted, 0, 1)
        
        return contrasted


def create_augmentation_pipeline(augmentation_type: str = "basic") -> DataAugmentation:
    """
    创建数据增强管道
    
    Args:
        augmentation_type: 增强类型 ("basic" 或 "advanced")
        
    Returns:
        数据增强对象
    """
    if augmentation_type == "basic":
        return DataAugmentation(
            rotation_range=5.0,
            scale_range=(0.95, 1.05),
            noise_std=0.01,
            brightness_range=(0.9, 1.1),
            apply_prob=0.5
        )
    elif augmentation_type == "advanced":
        return AdvancedDataAugmentation(
            rotation_range=5.0,
            scale_range=(0.95, 1.05),
            noise_std=0.01,
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            apply_prob=0.7
        )
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


# 测试函数
def test_augmentation():
    """测试数据增强效果"""
    # 创建测试图像
    test_image = torch.randn(1, 64, 64)
    
    # 创建增强管道
    aug = create_augmentation_pipeline("advanced")
    
    # 应用增强
    augmented = aug(test_image)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Original range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # 测试多次增强
    print("\n测试多次增强效果:")
    for i in range(5):
        aug_result = aug(test_image)
        print(f"增强 {i+1}: 范围 [{aug_result.min():.3f}, {aug_result.max():.3f}]")
    
    print("\n✅ 数据增强测试完成！")


if __name__ == "__main__":
    test_augmentation()
