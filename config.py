from pathlib import Path


class Configs(object):
    def __init__(self, gpus_num):
        super(Configs, self).__init__()
        # data configs
        self.rgbd_dirs = Path("RGBD_Datasets")  # RGBD data path
        self.hole_dirs = Path("Hole_Datasets")  # Hole data path
        self.dataset_dir = Path("Private_Real_Dataset")  # Private Real Dataset path (chứa Rgb/, Depth/, Lidar/)
        self.save_dir = Path("checkpoints")
        self.checkpoint = None  # checkpoint path

        # dataloader configs
        self.sizes = 320  # sizes of images during training

        # optimizer setting
        self.lr = 5e-5  # learning rate
        self.wd = 0.05  # weight decay
        self.epochs = 5  # epochs numbers
        self.batch_size = 32 // gpus_num  # batch sizes*4GPU

        # multi GPU and AMP
        self.num_workers = 4  # the number of workers
        self.amp = True  # automatic mixed precision (AMP)

        # feedback
        self.feedback_iteration = 50  # In loss thường xuyên hơn (dataset nhỏ 460 mẫu)
        self.checkpoint_epoch = 1  # Lưu checkpoint mỗi epoch
