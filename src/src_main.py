import datetime
import time
import timeit

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import save_img, min_max_norm, print_model_parm_nums, StandardizeData
from .data_tools import get_dataloader
from .data_tools_real import get_real_dataloader
from .losses import (
    WeightedDataLoss,
    WeightedMSGradLoss,
)
from .networks import UNet


class G2_MonoDepth:
    def __init__(self, cf, rank):
        self.configs = cf
        self.rank = rank
        self.network = UNet(rezero=True).cuda().train()
        if rank == 0:
            print_model_parm_nums(self.network)  # the number of parameters
        self.network = DDP(
            self.network, device_ids=[self.rank], static_graph=True
        )  # Use DistributedDataParallel:
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.network.parameters(),
                    "initial_lr": cf.lr,
                }
            ],
            lr=cf.lr,
            weight_decay=cf.wd,
        )  # create optimizers
        # dataloader and datasampler
        # Nếu có dataset_dir (Private_Real_Dataset) → dùng real dataloader
        # Nếu không → dùng dataloader gốc (RGBD_Datasets + Hole_Datasets)
        if hasattr(cf, 'dataset_dir') and cf.dataset_dir is not None:
            self.loader, self.sampler = get_real_dataloader(
                cf.dataset_dir,
                cf.batch_size,
                cf.sizes,
                self.rank,
                cf.num_workers,
            )
        else:
            self.loader, self.sampler = get_dataloader(
                cf.rgbd_dirs,
                cf.hole_dirs,
                cf.batch_size,
                cf.sizes,
                self.rank,
                cf.num_workers,
            )
        self.iteration_num = len(self.loader)
        # learning rate scheduler: cosine decay
        self.scheduler = CosineAnnealingLR(
            self.optimizer, self.iteration_num * cf.epochs
        )
        # use amp
        self.scaler = GradScaler('cuda') if cf.amp else None
        self.start_epoch = 0
        # resume train / load pretrained weights
        if cf.checkpoint is not None:
            if self.rank == 0:
                mode_str = "fine-tuning (load weights only)" if getattr(cf, 'finetune', False) else "resuming training"
                print(f"Loading checkpoint for {mode_str}...")
            # load checkpoint
            map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
            checkpoint = torch.load(cf.checkpoint, map_location=map_location)

            # Load trọng số vào module của DDP
            pretrained_dict = checkpoint["network"]
            model_dict = self.network.module.state_dict()
            # Chỉ load những lớp có tên và kích thước giống nhau (bỏ qua Head đã sửa)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            if self.rank == 0:
                print(f"Đã load {len(pretrained_dict)}/{len(model_dict)} lớp từ checkpoint.")
            model_dict.update(pretrained_dict)
            self.network.module.load_state_dict(model_dict)

            # Nếu resume training (không phải finetune) → load optimizer/scheduler/scaler và tiếp tục epoch
            if not getattr(cf, 'finetune', False):
                self.start_epoch = checkpoint["epoch"]
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.scaler is not None and "scaler" in checkpoint:
                    self.scaler.load_state_dict(checkpoint["scaler"])
            else:
                if self.rank == 0:
                    print(f"Fine-tune mode: bắt đầu từ epoch 1, lr={cf.lr}")

        # pytorch 2.0 compile (sau khi load weights để tránh xung đột)
        self.network = torch.compile(self.network)

        self.reg_function = WeightedDataLoss()
        self.sta_tool = StandardizeData()
        self.grad_function = WeightedMSGradLoss()

    def optimize_one_iteration(self, rgb, gt, raw):
        hole_gt = torch.where(gt == 0, torch.zeros_like(gt), torch.ones_like(gt))
        hole_raw = torch.where(raw == 0, torch.zeros_like(raw), torch.ones_like(raw))

        with autocast('cuda', enabled=(self.scaler is not None)):
            # loss in absolute domain
            depth = self.network(rgb, raw, hole_raw)
            loss_adepth = self.reg_function(depth, gt, hole_raw)
            # loss in relative domain
            sta_depth, sta_gt = self.sta_tool(depth, gt, hole_gt)
            loss_rdepth = self.reg_function(sta_depth, sta_gt, hole_gt)
            # sobel grad
            loss_rgrad = self.grad_function(sta_depth, sta_gt, hole_gt)

            loss = 1.5 * loss_adepth + loss_rdepth + 0.8 * loss_rgrad

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return loss, loss_adepth, loss_rdepth, loss_rgrad, depth

    def feedback_module(self, elapsed, epoch, iter_step, loss, loss_adepth, loss_rdepth, loss_rgrad, lr, summary, global_step):
        print(
            "Epoch:[%d]|batch:[%d/%d]|lr:%.2e|loss:%.4f|adepth:%.4f|rdepth:%.4f|rgrad:%.4f|elapsed:%s"
            % (
                epoch,
                iter_step,
                self.iteration_num,
                lr,
                float(loss),
                float(loss_adepth),
                float(loss_rdepth),
                float(loss_rgrad),
                elapsed,
            )
        )
        # log loss components
        summary.add_scalar("loss/loss", loss, global_step=global_step)
        summary.add_scalar("loss/loss_adepth", loss_adepth, global_step=global_step)
        summary.add_scalar("loss/loss_rdepth", loss_rdepth, global_step=global_step)
        summary.add_scalar("loss/loss_rgrad", loss_rgrad, global_step=global_step)
        summary.add_scalar("lr/lr", lr, global_step=global_step)

    @staticmethod
    def save_imgs(rgb, gt, raw, pred, log_dir, epoch, iter_step):
        # make dir
        epoch_dir = log_dir / ("epoch_" + str(epoch))
        epoch_dir.mkdir(exist_ok=True)
        file_last = f"_{epoch}_{iter_step}.png"
        # save the images:
        save_img(rgb[0], epoch_dir / ("rgb" + file_last))
        save_img(gt[0], epoch_dir / ("gt" + file_last))
        save_img(raw[0], epoch_dir / ("raw" + file_last))
        save_img(pred[0], epoch_dir / ("pred" + file_last))
        save_img(min_max_norm(gt[0]), epoch_dir / ("norm_gt" + file_last))
        save_img(min_max_norm(pred[0]), epoch_dir / ("norm_pred" + file_last))

    def train(self, cf):
        if self.rank == 0:
            # model/log rgbd_dirs
            model_dir, log_dir = cf.save_dir / "models", cf.save_dir / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            # tensorboard summarywriter
            summary = SummaryWriter(str(log_dir / "tensorboard"))
            # create a global time counter
            global_time = time.time()
            print("Starting the training process ... ")
        # epoch start
        global_step = self.start_epoch * self.iteration_num
        for epoch in range(self.start_epoch + 1, cf.epochs + 1):
            # set epoch in Distributed samplers
            self.sampler.set_epoch(epoch)
            # record time at the start of epoch
            if self.rank == 0:
                start = timeit.default_timer()
                print(f"\nEpoch: [{epoch}/{cf.epochs}]")
                epoch_loss_sum = 0.0
                epoch_adepth_sum = 0.0
                epoch_rdepth_sum = 0.0
                epoch_rgrad_sum = 0.0
                epoch_count = 0
            for i, (rgb, gt, raw) in enumerate(self.loader, start=1):
                # get data
                rgb = rgb.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                raw = raw.cuda(non_blocking=True)
                # optimizing
                loss, loss_adepth, loss_rdepth, loss_rgrad, pred = self.optimize_one_iteration(rgb, gt, raw)
                self.scheduler.step()  # learning rate decay
                # Tích lũy loss cho epoch summary
                if self.rank == 0:
                    epoch_loss_sum += float(loss)
                    epoch_adepth_sum += float(loss_adepth)
                    epoch_rdepth_sum += float(loss_rdepth)
                    epoch_rgrad_sum += float(loss_rgrad)
                    epoch_count += 1
                # logging
                global_step += 1
                effective_feedback = 1 if self.iteration_num <= cf.feedback_iteration else cf.feedback_iteration
                if self.rank == 0 and ((i % effective_feedback == 0) or (i == 1) or (i == self.iteration_num)):
                    with torch.no_grad():
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        lr = self.optimizer.param_groups[0]["lr"]
                        # log and print
                        self.feedback_module(
                            elapsed,
                            epoch,
                            i,
                            loss,
                            loss_adepth,
                            loss_rdepth,
                            loss_rgrad,
                            lr,
                            summary,
                            global_step,
                        )
                        # save intermediate results
                        self.save_imgs(rgb, gt, raw, pred, log_dir, epoch, i)
            # epoch summary & logging checkpoint
            if self.rank == 0:
                stop = timeit.default_timer()
                avg_loss = epoch_loss_sum / max(epoch_count, 1)
                avg_adepth = epoch_adepth_sum / max(epoch_count, 1)
                avg_rdepth = epoch_rdepth_sum / max(epoch_count, 1)
                avg_rgrad = epoch_rgrad_sum / max(epoch_count, 1)
                print(
                    f"── Epoch [{epoch}/{cf.epochs}] Summary ──\n"
                    f"  avg_loss={avg_loss:.4f} | avg_adepth={avg_adepth:.4f} | "
                    f"avg_rdepth={avg_rdepth:.4f} | avg_rgrad={avg_rgrad:.4f}\n"
                    f"  time={stop - start:.3f}s"
                )
                summary.add_scalar("epoch/avg_loss", avg_loss, global_step=epoch)
                summary.add_scalar("epoch/avg_adepth", avg_adepth, global_step=epoch)
                summary.add_scalar("epoch/avg_rdepth", avg_rdepth, global_step=epoch)
                summary.add_scalar("epoch/avg_rgrad", avg_rgrad, global_step=epoch)
                if epoch % cf.checkpoint_epoch == 0 or epoch == cf.epochs:
                    save_file = model_dir / f"epoch_{epoch}.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "network": self.network.module.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "scaler": self.scaler.state_dict(),
                        },
                        save_file,
                    )

        if self.rank == 0:
            print("Training completed ...")
