"""Script chẩn đoán: kiểm tra giá trị pixel của raw_1%, gt, và result."""
import numpy as np
from PIL import Image
from pathlib import Path

test_dir = Path("Test_Datasets/Private_Real")

# Lấy 1 file mẫu
sample_id = "1"
raw_path = test_dir / "raw_1%" / f"{sample_id}.png"
gt_path = test_dir / "gt" / f"{sample_id}.png"
rgb_path = test_dir / "rgb" / f"{sample_id}.png"
result_path = test_dir / "result_ONNX1%" / f"{sample_id}.png"

print("=" * 60)
print(f"KIỂM TRA MẪU: ID = {sample_id}")
print("=" * 60)

# 1. Kiểm tra RGB
if rgb_path.exists():
    rgb = np.array(Image.open(rgb_path))
    print(f"\n[RGB] {rgb_path}")
    print(f"  Shape: {rgb.shape}, Dtype: {rgb.dtype}")
    print(f"  Min: {rgb.min()}, Max: {rgb.max()}")

# 2. Kiểm tra GT depth
if gt_path.exists():
    gt = np.array(Image.open(gt_path))
    print(f"\n[GT Depth] {gt_path}")
    print(f"  Shape: {gt.shape}, Dtype: {gt.dtype}")
    print(f"  Min: {gt.min()}, Max: {gt.max()}")
    print(f"  Mean (non-zero): {gt[gt > 0].mean():.1f}")
    print(f"  Pixels > 0: {np.count_nonzero(gt)} / {gt.size} ({100*np.count_nonzero(gt)/gt.size:.1f}%)")

# 3. Kiểm tra Sparse depth (raw_1%)
if raw_path.exists():
    raw = np.array(Image.open(raw_path))
    print(f"\n[Sparse Depth - raw_1%] {raw_path}")
    print(f"  Shape: {raw.shape}, Dtype: {raw.dtype}")
    print(f"  Min: {raw.min()}, Max: {raw.max()}")
    nonzero = np.count_nonzero(raw)
    print(f"  Pixels > 0: {nonzero} / {raw.size} ({100*nonzero/raw.size:.4f}%)")
    if nonzero > 0:
        print(f"  Mean (non-zero): {raw[raw > 0].mean():.1f}")
        # Hiển thị vị trí các điểm LiDAR
        ys, xs = np.where(raw > 0)
        print(f"  Tọa độ Y range: [{ys.min()}, {ys.max()}]")
        print(f"  Tọa độ X range: [{xs.min()}, {xs.max()}]")
else:
    print(f"\n[Sparse Depth] KHÔNG TÌM THẤY: {raw_path}")

# 4. Kiểm tra Result
if result_path.exists():
    result = np.array(Image.open(result_path))
    print(f"\n[Result ONNX] {result_path}")
    print(f"  Shape: {result.shape}, Dtype: {result.dtype}")
    print(f"  Min: {result.min()}, Max: {result.max()}")
    print(f"  Mean: {result.mean():.1f}")
    
    # So sánh scale với GT
    if gt_path.exists():
        gt_f = gt.astype(np.float32)
        res_f = result.astype(np.float32)
        gt_f[gt_f == 0] = np.nan
        res_f_masked = res_f.copy()
        res_f_masked[np.isnan(gt_f)] = np.nan
        print(f"\n[SO SÁNH SCALE]")
        print(f"  GT mean (valid): {np.nanmean(gt_f):.1f}")
        print(f"  Result mean (at valid GT): {np.nanmean(res_f_masked):.1f}")
        print(f"  Ratio (Result/GT): {np.nanmean(res_f_masked) / np.nanmean(gt_f):.3f}")
else:
    print(f"\n[Result] KHÔNG TÌM THẤY: {result_path}")

# 5. Kiểm tra thêm vài mẫu
print(f"\n{'=' * 60}")
print("KIỂM TRA NHANH 5 MẪU ĐẦU")
print("=" * 60)
for sid in ["1", "2", "3", "4", "5"]:
    rp = test_dir / "raw_1%" / f"{sid}.png"
    gp = test_dir / "gt" / f"{sid}.png"
    ep = test_dir / "result_ONNX1%" / f"{sid}.png"
    if rp.exists() and gp.exists():
        r = np.array(Image.open(rp))
        g = np.array(Image.open(gp))
        nz = np.count_nonzero(r)
        line = f"  ID={sid}: raw_nz={nz:3d}"
        line += f", gt_range=[{g.min()}-{g.max()}]"
        if ep.exists():
            e = np.array(Image.open(ep))
            line += f", result_range=[{e.min()}-{e.max()}]"
        print(line)
