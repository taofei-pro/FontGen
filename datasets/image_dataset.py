from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from utils.image.image_utils import check_names_match, get_image_names, get_image_paths
from utils.data_augmentation import create_augmentation_pipeline


def create_transform(
    normalize: bool,
) -> T.Compose:
    """
    Create a transform for the dataset.
    """

    transform_list = [T.ToTensor()]

    if normalize:
        transform_list.append(T.Normalize(mean=(0.5,), std=(0.5,)))

    return T.Compose(transform_list)


class PairedGlyphImageDataset(Dataset):
    """
    Paired image dataset for training LDM.
    """

    def __init__(
        self,
        target_img_dir: str | Path,
        reference_img_dir: str | Path,
        use_data_augmentation: bool = False,
        augmentation_type: str = "basic",
        rotation_range: float = 5.0,
        scale_range: tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.01,
        brightness_range: tuple[float, float] = (0.9, 1.1),
        contrast_range: tuple[float, float] = (0.9, 1.1),
        augmentation_prob: float = 0.5,
    ):
        self.tgt_img_paths = get_image_paths(target_img_dir)
        self.ref_img_paths = get_image_paths(reference_img_dir)

        check_names_match(self.tgt_img_paths, self.ref_img_paths)
        self.img_names = get_image_names(self.tgt_img_paths)

        self.transform = create_transform(
            normalize=True,
        )
        
        # 数据增强设置
        self.use_data_augmentation = use_data_augmentation
        if use_data_augmentation:
            self.augmentation = create_augmentation_pipeline(augmentation_type)
            print(f"✅ 数据增强已启用: {augmentation_type} 模式")
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.ref_img_paths)

    def __getitem__(self, idx):
        tgt_img = Image.open(self.tgt_img_paths[idx]).convert("L")
        ref_img = Image.open(self.ref_img_paths[idx]).convert("L")

        tgt_img = self.transform(tgt_img)
        ref_img = self.transform(ref_img)
        
        # 应用数据增强（仅对训练数据）
        if self.use_data_augmentation and self.augmentation is not None:
            tgt_img = self.augmentation(tgt_img)
            ref_img = self.augmentation(ref_img)

        img_name = self.img_names[idx]

        return {"tgt_img": tgt_img, "ref_img": ref_img, "img_name": img_name}

    def get_img_names(self) -> list[str]:
        """
        Get image names from the dataset.
        """
        return self.img_names

    def get_img_name_at_index(
        self,
        idx: int,
    ) -> str:
        """
        Get the image name at a specific index.
        """
        return self.img_names[idx]


class MetricsImageDataset(Dataset):
    """
    Paired image dataset for computing metrics.
    """

    def __init__(
        self,
        generated_img_dir: str | Path,
        ground_truth_img_dir: str | Path,
        normalize: bool,
        convert_to_rgb: bool,
    ):
        self.gen_img_paths = get_image_paths(generated_img_dir)
        self.gt_img_paths = get_image_paths(ground_truth_img_dir)

        self.normalize = normalize
        self.convert_to_rgb = convert_to_rgb

        check_names_match(self.gen_img_paths, self.gt_img_paths)
        self.img_names = get_image_names(self.gen_img_paths)

        self.transform = create_transform(
            normalize=normalize,
        )

    def __len__(self):
        return len(self.gen_img_paths)

    def __getitem__(self, idx):
        gen_img = Image.open(self.gen_img_paths[idx]).convert("L")
        gt_img = Image.open(self.gt_img_paths[idx]).convert("L")

        gen_img = self.transform(gen_img)
        gt_img = self.transform(gt_img)

        if self.convert_to_rgb:
            gen_img = gen_img.repeat(3, 1, 1)
            gt_img = gt_img.repeat(3, 1, 1)

        return {
            "gen_img": gen_img,
            "gt_img": gt_img,
        }


class GlyphImageDataset(Dataset):
    """
    Single image dataset for inference/utility pipelines.
    """

    def __init__(
        self,
        img_dir: str | Path,
        normalize: bool = True,
    ):
        self.img_paths = get_image_paths(img_dir)
        self.img_names = get_image_names(self.img_paths)
        self.transform = create_transform(normalize=normalize)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.img_paths[idx]).convert("L")
        img = self.transform(img)
        img_name = self.img_names[idx]
        return {"tgt_img": img, "img_name": img_name}