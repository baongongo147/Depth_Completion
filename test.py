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
        default="Test_Datasets/Ibims",
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--model_dir",
        type=lambda x: Path(x),
        default="checkpoints/models/epoch_100.pth",
        help="Path to load models",
    )
    args = parser.parse_args()
    return args


def demo_save(args):
    print("-----------building model-------------")
    network = UNet(rezero=True).to(device).eval()
    network.load_state_dict(torch.load(args.model_dir, map_location=device)["network"])
    raw_dirs = ["0%", "1%", "100%"]
    print("-----------inferring---------------")
    for raw_dir in raw_dirs:
        with torch.no_grad():
            for file in (args.rgbd_dir / "rgb").rglob("*.png"):
                str_file = str(file)
                raw_path = str_file.replace(os.sep + "rgb" + os.sep, os.sep + "raw_" + raw_dir + os.sep)
                save_path = str_file.replace(os.sep + "rgb" + os.sep, os.sep + "result_" + raw_dir + os.sep)
                rgbd_reader = RGBPReader()
                # processing
                rgb, raw, hole_raw = rgbd_reader.read_data(str_file, raw_path)
                if raw.dim() == 5:
                    raw = raw[:, :, :, :, 0] # Lấy kênh đầu tiên trong 3 kênh màu
                if hole_raw.dim() == 5:
                    hole_raw = hole_raw[:, :, :, :, 0]
                pred = network(rgb.to(device), raw.to(device), hole_raw.to(device))
                pred = rgbd_reader.adjust_domain(pred)
                # # save img
                os.makedirs(str(Path(save_path).parent), exist_ok=True)
                Image.fromarray(pred).save(save_path)
                print(raw_path)


def demo_metric(args):
    raw_dirs = ["0%", "1%", "100%"]
    for raw_dir in raw_dirs:
        srmse = 0.0
        ord_error = 0.0
        rmse = 0.0
        rel = 0.0
        count = 0.0
        for file in (args.rgbd_dir / "rgb").rglob("*.png"):
            count += 1.0

            str_file = str(file)
            pred_path = str_file.replace("/rgb/", "/result_" + raw_dir + "/")
            gt_path = str_file.replace("/rgb/", "/gt/")
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