"""
Dataset loader cho Private_Real_Dataset.
Sử dụng tọa độ pixel từ LiDAR thật để tạo sparse depth input,
thay vì mô phỏng bằng cách lấy mẫu đều trên 1 dòng từ dense depth.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as trans
import torchvision.transforms.functional as TF

from .data_tools import rgb_read, depth_read, RandomDepth


class RealResizedCropRGBDR(trans.RandomResizedCrop):
    """
    RandomResizedCrop cho 5 channels: RGB(3) + GT(1) + Raw(1).
    Theo dõi hole mask cho cả GT và Raw để tránh artifact khi resize interpolation.
    """
    def forward(self, img):
        # img: 5 channels [rgb(3), gt(1), raw(1)]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # Tạo mask (non-zero pixels) cho gt và raw trước khi resize
        gt_mask = (img[3:4, :, :] != 0).float()
        raw_mask = (img[4:5, :, :] != 0).float()

        # Concat: [rgb(3), gt(1), raw(1), gt_mask(1), raw_mask(1)] = 7 channels
        all_ch = torch.cat([img, gt_mask, raw_mask], dim=0)
        all_ch = TF.resized_crop(
            all_ch, i, j, h, w, self.size,
            self.interpolation, antialias=self.antialias
        )

        # Sau resize, áp lại mask để loại bỏ giá trị giả do interpolation
        gt_mask = (all_ch[5, :, :] >= 0.99).float()
        raw_mask = (all_ch[6, :, :] >= 0.99).float()

        rgb = all_ch[:3, :, :]
        gt = (all_ch[3, :, :] * gt_mask).unsqueeze(0)
        raw = (all_ch[4, :, :] * raw_mask).unsqueeze(0)

        return torch.cat([rgb, gt, raw], dim=0)


class RealTransformUtils(object):
    """
    Transform pipeline cho dữ liệu thực tế.
    Xử lý 5 channels (rgb + gt + raw) cùng nhau để giữ đồng bộ spatial transforms.
    """
    def __init__(self, size):
        # Spatial transforms cho cả 5 channels
        self._spatial_transform = trans.Compose([
            trans.RandomCrop(size),
            RealResizedCropRGBDR(size, (0.64, 1.0), antialias=True),
            trans.RandomHorizontalFlip(0.5),
        ])
        # Augmentation riêng cho RGB (color) và GT (depth scale)
        self._rgb_transform = trans.ColorJitter(0.2, 0.2, 0.2)
        self._gt_transform = RandomDepth(0.2)

    def trans_rgbgtraw(self, rgb: Tensor, gt: Tensor, raw: Tensor):
        """Transform RGB, GT và Raw sparse cùng nhau."""
        # Concat 5 channels và áp spatial transforms
        combined = self._spatial_transform(torch.cat([rgb, gt, raw], dim=0))
        rgb = combined[:3, :, :]
        gt = combined[3, :, :].unsqueeze(0)
        raw = combined[4, :, :].unsqueeze(0)

        # Augmentation riêng
        rgb = self._rgb_transform(rgb)
        gt = self._gt_transform(gt)

        return rgb, gt, raw


class PrivateRealDataset(Dataset):
    """
    Dataset cho Private_Real_Dataset.

    Cấu trúc thư mục:
        Private_Real_Dataset/
        ├── Rgb/           → {id}_rgb.png
        ├── Depth/         → {id}_depth_16bit.png
        ├── Lidar/         → {id}_lidar.json
        └── Rgb-Fusion/    → {id}_rgb_fusion.png (không dùng)

    Thay vì mô phỏng sparse LiDAR bằng cách lấy 1 dòng đều từ dense depth,
    dataset này sử dụng tọa độ pixel LiDAR thật từ JSON để tạo sparse depth input.
    """
    def __init__(self, dataset_dir, size, min_lidar_points=5):
        """
        Args:
            dataset_dir: Path tới thư mục Private_Real_Dataset
            size: Kích thước ảnh khi train (e.g. 320)
            min_lidar_points: Số điểm LiDAR tối thiểu sau transform,
                              nếu ít hơn sẽ fallback sang sampling từ GT
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.size = size
        self.min_lidar_points = min_lidar_points
        self.transforms = RealTransformUtils(size)

        # Quét tất cả bộ 3 file (rgb, depth_16bit, lidar)
        self.samples = []
        rgb_dir = self.dataset_dir / "Rgb"
        for rgb_file in sorted(rgb_dir.glob("*_rgb.png")):
            file_id = rgb_file.stem.replace("_rgb", "")
            depth_file = self.dataset_dir / "Depth" / f"{file_id}_depth_16bit.png"
            lidar_file = self.dataset_dir / "Lidar" / f"{file_id}_lidar.json"

            if depth_file.exists() and lidar_file.exists():
                self.samples.append({
                    "id": file_id,
                    "rgb": rgb_file,
                    "depth": depth_file,
                    "lidar": lidar_file,
                })

        print(f"[PrivateRealDataset] Loaded {len(self.samples)} samples")

    def _create_sparse_from_lidar(self, gt, lidar_path):
        """
        Tạo sparse depth map bằng cách lấy giá trị GT tại các tọa độ pixel LiDAR thật.

        Cách này kết hợp ưu điểm của cả hai:
        - Vị trí pixel: Từ LiDAR thật (pattern quét không đều, có gaps thực tế)
        - Giá trị depth: Từ depth camera (đảm bảo đồng nhất scale với GT)

        Args:
            gt: Tensor (1, H, W) - ground truth depth đã normalize [0, 1]
            lidar_path: Path tới file JSON chứa dữ liệu LiDAR

        Returns:
            Tensor (1, H, W) - sparse depth map
        """
        _, h, w = gt.shape
        sparse = torch.zeros_like(gt)

        with open(lidar_path, 'r') as f:
            lidar_data = json.load(f)

        pixel_points = lidar_data["labels"]["image_pixel_points"]

        for (px, py) in pixel_points:
            # Kiểm tra tọa độ nằm trong ảnh
            if 0 <= px < w and 0 <= py < h:
                # Lấy giá trị depth từ GT tại vị trí LiDAR thật
                depth_val = gt[0, py, px].item()
                if depth_val > 0:
                    sparse[0, py, px] = depth_val

        return sparse

    @staticmethod
    def _fallback_lidar_line(gt, step=5):
        """
        Fallback: Nếu sau transform, sparse depth mất hết điểm LiDAR,
        lấy mẫu 1 dòng từ GT (giống cách cũ nhưng không thêm noise).
        """
        _, h, w = gt.shape
        raw = torch.zeros_like(gt)
        target_row = np.random.randint(max(0, h // 2 - 20), min(h, h // 2 + 20))
        for col in range(0, w, step):
            raw[0, target_row, col] = gt[0, target_row, col]
        return raw

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Đọc RGB và Ground Truth depth
        rgb = rgb_read(sample["rgb"])       # (3, H, W), [0, 1]
        gt = depth_read(sample["depth"])    # (1, H, W), [0, 1]

        # 2. Tạo sparse depth từ LiDAR thật (tại tọa độ pixel gốc)
        raw = self._create_sparse_from_lidar(gt, sample["lidar"])

        # 3. Transform cả 3 cùng nhau (crop, resize, flip)
        #    Đảm bảo tọa độ spatial đồng bộ
        rgb, gt, raw = self.transforms.trans_rgbgtraw(rgb, gt, raw)

        # 4. Fallback nếu crop/resize làm mất hết điểm LiDAR
        if raw.sum() == 0 or (raw != 0).sum() < self.min_lidar_points:
            raw = self._fallback_lidar_line(gt)

        return rgb, gt, raw


def get_real_dataloader(dataset_dir, batch_size, sizes, rank, num_workers):
    """Tạo DataLoader cho Private_Real_Dataset."""
    dataset = PrivateRealDataset(dataset_dir, sizes)
    if rank == 0:
        print(f"Loaded the real dataset with: {len(dataset)} images...\n")

    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, sampler
