import os
import torch
import argparse
from pathlib import Path
from src.src_main import G2_MonoDepth
from src.utils import DDPutils
from config import Configs


# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thay đổi các đường dẫn này cho khớp với Dataset 
def parse_arguments():
    parser = argparse.ArgumentParser(
        "Fine-tuning options for G2-MonoDepth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # config dataset and pretrained model path
    parser.add_argument(
        "--rgbd_dir",
        type=lambda x: Path(x),
        default=Path("RGBD_Datasets"),
        help="Path to RGBD folder",
    )
    parser.add_argument(
        "--hole_dir",
        type=lambda x: Path(x),
        default=Path("Hole_Datasets"),
        help="Path to Hole masks folder",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/models/epoch_100.pth",
        help="Path to load pre-trained weights",
    )
    parser.add_argument(
        "--save_dir",
        type=lambda x: Path(x),
        default=Path("checkpoints_finetune"),
        help="Directory to save fine-tuned models",
    )
    args = parser.parse_args()
    return args

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Sử dụng 2 GPU T4 trên Kaggle nếu có

def DDP_finetune(rank, world_size, args):
    # 1. Khởi tạo Config
    cf = Configs(world_size)
    
    # 2. Ghi đè cấu hình để Fine-tune
    cf.rgbd_dirs = args.rgbd_dir
    cf.hole_dirs = args.hole_dir
    cf.save_dir = args.save_dir
    cf.checkpoint = args.model_dir  # Load trọng số cũ

    # 3. Thiết lập DDP
    DDPutils.setup(rank, world_size, 6003)
    
    if rank == 0:
        print(f"--- BẮT ĐẦU FINE-TUNING ---")
        print(f"Dữ liệu: {cf.rgbd_dirs}")
        print(f"Trọng số gốc: {cf.checkpoint}")
        if not args.save_dir.exists():
            args.save_dir.mkdir(parents=True)

    # 4. Khởi tạo Trainer
    # Lưu ý: Class G2_MonoDepth của bạn đã có logic load checkpoint trong __init__
    trainer = G2_MonoDepth(cf, rank=rank)
    
    # 5. Bắt đầu Train
    trainer.train(cf)
    
    DDPutils.cleanup()

if __name__ == "__main__":
    args = parse_arguments()

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"[INFO] Phát hiện {n_gpus} GPU. Đang khởi động Fine-tune...")
        DDPutils.run_demo(lambda r, w: DDP_finetune(r, w, args), n_gpus)
    else:
        print("[ERROR] Fine-tune yêu cầu GPU có CUDA!")