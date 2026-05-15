"""
Script chuyển đổi Private_Real_Dataset sang format Test_Datasets
để test với test_ONNX.py.

Cấu trúc đầu vào (Private_Real_Dataset):
    Rgb/{id}_rgb.png
    Depth/{id}_depth_16bit.png
    Lidar/{id}_lidar.json

Cấu trúc đầu ra (Test_Datasets/Private_Real):
    rgb/{id}.png            ← copy từ Rgb/{id}_rgb.png
    raw_1%/{id}.png         ← sparse depth từ LiDAR JSON (16-bit PNG)
    gt/{id}.png             ← copy từ Depth/{id}_depth_16bit.png
"""
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

SRC_DIR = Path("Private_Real_Dataset")
DST_DIR = Path("Test_Datasets/Private_Real")

def create_sparse_depth_from_lidar(depth_path, lidar_path, output_path):
    """
    Tạo sparse depth 16-bit PNG từ tọa độ LiDAR thật.
    Tại mỗi pixel LiDAR, lấy giá trị từ depth ground truth.
    """
    # Đọc depth ground truth
    depth_img = Image.open(depth_path)
    depth_np = np.array(depth_img)  # 16-bit values
    h, w = depth_np.shape[:2]

    # Tạo sparse depth (toàn 0)
    sparse = np.zeros((h, w), dtype=depth_np.dtype)

    # Đọc LiDAR JSON
    with open(lidar_path, 'r') as f:
        lidar_data = json.load(f)

    pixel_points = lidar_data["labels"]["image_pixel_points"]

    # Tại mỗi tọa độ LiDAR, copy giá trị depth từ GT
    point_count = 0
    for (px, py) in pixel_points:
        if 0 <= px < w and 0 <= py < h:
            val = depth_np[py, px] if depth_np.ndim == 2 else depth_np[py, px, 0]
            if val > 0:
                sparse[py, px] = val
                point_count += 1

    # Lưu dưới dạng 16-bit PNG
    sparse_img = Image.fromarray(sparse.astype(np.uint16))
    sparse_img.save(output_path)
    return point_count


def main():
    # Tạo thư mục đích
    rgb_dst = DST_DIR / "rgb"
    raw_dst = DST_DIR / "raw_1%"
    gt_dst = DST_DIR / "gt"

    rgb_dst.mkdir(parents=True, exist_ok=True)
    raw_dst.mkdir(parents=True, exist_ok=True)
    gt_dst.mkdir(parents=True, exist_ok=True)

    # Quét tất cả file RGB
    rgb_files = sorted(SRC_DIR.glob("Rgb/*_rgb.png"))
    print(f"Tìm thấy {len(rgb_files)} ảnh RGB trong {SRC_DIR}/Rgb/")

    success = 0
    for rgb_file in rgb_files:
        file_id = rgb_file.stem.replace("_rgb", "")
        depth_file = SRC_DIR / "Depth" / f"{file_id}_depth_16bit.png"
        lidar_file = SRC_DIR / "Lidar" / f"{file_id}_lidar.json"

        if not depth_file.exists():
            print(f"  ✗ Thiếu depth: {depth_file}")
            continue
        if not lidar_file.exists():
            print(f"  ✗ Thiếu lidar: {lidar_file}")
            continue

        # 1. Copy RGB
        shutil.copy2(rgb_file, rgb_dst / f"{file_id}.png")

        # 2. Copy GT depth (16-bit)
        shutil.copy2(depth_file, gt_dst / f"{file_id}.png")

        # 3. Tạo sparse depth từ LiDAR thật
        n_points = create_sparse_depth_from_lidar(
            depth_file, lidar_file, raw_dst / f"{file_id}.png"
        )

        success += 1
        if success % 50 == 0:
            print(f"  ✓ Đã xử lý {success}/{len(rgb_files)} ảnh...")

    print(f"\n{'='*50}")
    print(f"Hoàn tất! {success}/{len(rgb_files)} ảnh đã được copy.")
    print(f"Cấu trúc thư mục:")
    print(f"  {DST_DIR}/rgb/       → {len(list(rgb_dst.glob('*.png')))} ảnh RGB")
    print(f"  {DST_DIR}/raw_1%/    → {len(list(raw_dst.glob('*.png')))} ảnh sparse depth (LiDAR thật)")
    print(f"  {DST_DIR}/gt/        → {len(list(gt_dst.glob('*.png')))} ảnh GT depth")
    print(f"\nĐể test, chạy:")
    print(f'  python test_ONNX.py --rgbd_dir "{DST_DIR}"')


if __name__ == "__main__":
    main()
