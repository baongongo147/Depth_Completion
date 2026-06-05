import os
import argparse
import time
import numpy as np

from PIL import Image
from pathlib import Path
from src.networks import UNet
from test_utils import RGBPReader, DepthEvaluation
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus

# turn fast mode on
if device.type == "cuda":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for G2-MonoDepth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        type=lambda x: Path(x),
        default="Test_Datasets/Private_Test",
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--model_dir",
        type=lambda x: Path(x),
        default="checkpoints/models/epoch_30.pth",
        help="Path to load models",
    )
    args = parser.parse_args()
    return args


def pad_to_multiple(tensor, multiple=64):
    """Pad tensor (N,C,H,W) so H and W are multiples of `multiple`."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    # pad order: (left, right, top, bottom)
    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, h, w


def demo_save(args):
    print("-----------building model-------------")
    network = UNet(rezero=True).to(device).eval()
    network.load_state_dict(torch.load(args.model_dir, map_location=device)["network"])
    # raw_dirs = ["0%", "1%", "100%"]
    raw_dirs = ["1%"]
    print("-----------inferring---------------")
    # Prepare output folder based on model name
    model_name = Path(args.model_dir).stem
    base_save_dir = args.rgbd_dir / f"result_{model_name}"
    for raw_dir in raw_dirs:
        with torch.no_grad():
            for file in (args.rgbd_dir / "rgb").rglob("*.png"):
                str_file = str(file)
                # relative path under rgb folder
                rel_path = Path(file).relative_to(args.rgbd_dir / "rgb")
                raw_path = str_file.replace(os.sep + "rgb" + os.sep, os.sep + "raw_" + raw_dir + os.sep)
                save_path = str(base_save_dir / raw_dir / rel_path)
                rgbd_reader = RGBPReader()
                # processing
                rgb, raw, hole_raw = rgbd_reader.read_data(str_file, raw_path)
                if raw.dim() == 5:
                    raw = raw[:, :, :, :, 0] # Lấy kênh đầu tiên trong 3 kênh màu
                if hole_raw.dim() == 5:
                    hole_raw = hole_raw[:, :, :, :, 0]
                # Pad inputs to multiple of 64 for UNet compatibility
                rgb_pad, orig_h, orig_w = pad_to_multiple(rgb, 64)
                raw_pad, _, _ = pad_to_multiple(raw, 64)
                hole_raw_pad, _, _ = pad_to_multiple(hole_raw, 64)
                pred = network(rgb_pad.to(device), raw_pad.to(device), hole_raw_pad.to(device))
                # Crop back to original size
                pred = pred[:, :, :orig_h, :orig_w]
                pred = rgbd_reader.adjust_domain(pred)
                # # save img
                os.makedirs(str(Path(save_path).parent), exist_ok=True)
                Image.fromarray(pred).save(save_path)
                print(raw_path)


def demo_metric(args):
    # raw_dirs = ["0%", "1%", "100%"]
    raw_dirs = ["1%"]
    for raw_dir in raw_dirs:
        srmse = 0.0
        ord_error = 0.0
        rmse = 0.0
        rel = 0.0
        count = 0.0
        # Use same output folder naming as demo_save
        model_name = Path(args.model_dir).stem
        base_save_dir = args.rgbd_dir / f"result_{model_name}"

        for file in (args.rgbd_dir / "rgb").rglob("*.png"):
            count += 1.0

            str_file = str(file)
            rel_path = Path(file).relative_to(args.rgbd_dir / "rgb")

            pred_path = str(base_save_dir / raw_dir / rel_path)
            gt_path = str_file.replace(os.sep + "rgb" + os.sep, os.sep + "gt" + os.sep)
            # depth should be nonzero
            pred = np.clip(
                np.array(Image.open(pred_path)).astype(np.float32), 1.0, 65535.0
            )
            gt = np.array(Image.open(gt_path)).astype(np.float32)
            if raw_dir == "0%":
                srmse += DepthEvaluation.srmse(pred, gt)
                ord_error += DepthEvaluation.oe(pred, gt)
            else:
                rmse += DepthEvaluation.rmse(pred, gt)
                rel += DepthEvaluation.absRel(pred, gt)
        if raw_dir == "0%":
            srmse /= count
            ord_error /= count
            print("0%: srmse=", str(srmse), " oe=", str(ord_error))
        else:
            rmse /= count
            rel /= count
            print(raw_dir, ": rmse=", str(rmse), " Absrel=", str(rel))


if __name__ == "__main__":
    args = parse_arguments()
    start = time.time()
    demo_save(args)
    demo_metric(args)
    print(f"\nTotal time: {time.time() - start:.2f} seconds")