import os
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from src.networks import UNet
from test_utils import RGBPReader, DepthEvaluation
import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpus

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

TARGET_SIZE = (320, 320)


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


def resize_inputs(rgb, raw, hole_raw):
    """
    Resize inputs to 320x320 using correct interpolation.
    """
    rgb = F.interpolate(
        rgb, size=TARGET_SIZE, mode="bilinear", align_corners=False
    )
    raw = F.interpolate(
        raw, size=TARGET_SIZE, mode="nearest"
    )
    hole_raw = F.interpolate(
        hole_raw, size=TARGET_SIZE, mode="nearest"
    )
    return rgb, raw, hole_raw


def demo_save(args):
    print("-----------building model-------------")
    network = UNet(rezero=True).cuda().eval()
    network.load_state_dict(torch.load(args.model_dir)["network"])

    raw_dirs = ["1%"]
    print("-----------inferring---------------")

    rgbd_reader = RGBPReader()

    for raw_dir in raw_dirs:
        with torch.no_grad():
            for file in (args.rgbd_dir / "rgb").rglob("*.png"):
                str_file = str(file)
                raw_path = str_file.replace("/rgb/", "/raw_" + raw_dir + "/")
                save_path = str_file.replace("/rgb/", "/result_" + raw_dir + "/")

                # Load data
                rgb, raw, hole_raw = rgbd_reader.read_data(str_file, raw_path)

                # Resize to 320x320
                rgb, raw, hole_raw = resize_inputs(rgb, raw, hole_raw)

                # Forward
                pred = network(rgb.cuda(), raw.cuda(), hole_raw.cuda())
                pred = rgbd_reader.adjust_domain(pred)

                # Save
                os.makedirs(str(Path(save_path).parent), exist_ok=True)
                Image.fromarray(pred).save(save_path)

                print(raw_path)


def demo_metric(args):
    raw_dirs = ["1%"]

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

            # Load prediction
            pred = np.clip(
                np.array(Image.open(pred_path)).astype(np.float32),
                1.0,
                65535.0,
            )

            # Load + resize GT (NEAREST!)
            gt = np.array(Image.open(gt_path)).astype(np.float32)
            gt = np.array(
                Image.fromarray(gt).resize(TARGET_SIZE, resample=Image.NEAREST)
            )

            if raw_dir == "0%":
                srmse += DepthEvaluation.srmse(pred, gt)
                ord_error += DepthEvaluation.oe(pred, gt)
            else:
                rmse += DepthEvaluation.rmse(pred, gt)
                rel += DepthEvaluation.absRel(pred, gt)
                srmse += DepthEvaluation.srmse(pred, gt)
                ord_error += DepthEvaluation.oe(pred, gt)

        if raw_dir == "0%":
            srmse /= count
            ord_error /= count
            print("0%: srmse =", srmse, " oe =", ord_error)
        else:
            rmse /= count
            rel /= count
            print(raw_dir, ": rmse =", rmse, " AbsRel =", rel)
            srmse /= count
            ord_error /= count
            print("0%: srmse =", srmse, " oe =", ord_error)


if __name__ == "__main__":
    args = parse_arguments()
    demo_save(args)
    demo_metric(args)