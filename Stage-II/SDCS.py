import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sorcery import dict_of
import numpy as np
import os
import torchmetrics.functional.image as MF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import time

from network import SDCSNet
from schedulers import LinearWarmupCosineAnnealingLR
from util import check_and_make, regularize_inputs


class SDCSModel(pl.LightningModule):
    def __init__(self,
                 lr,
                 epochs,
                 bands,
                 rgb_c,
                 dataname,
                 ):
        super().__init__()
        self.automatic_optimization = False

        self.rgb_c = rgb_c
        self.model = SDCSNet(bands=bands)
        self.mse = torch.nn.MSELoss()
        self.dataname = dataname
        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(5)]

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        sche_opt = LinearWarmupCosineAnnealingLR(optimizer=opt_g, warmup_epochs=15, max_epochs=500, eta_min=5e-6)
        return [opt_g], [sche_opt]

    def forward(self, CSBS, LR_t2, HR_t1):
        pred = self.model(CSBS, LR_t2, HR_t1)
        out = dict_of(pred)
        return out

    def training_step(self, batch, batch_idx):
        CSBS, LR_t2, HR_t1, gt = batch["CSBS"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        self.model.zero_grad()
        opt = self.optimizers()
        out = self.forward(CSBS, LR_t2, HR_t1)
        pred = out["pred"]
        loss_g = self.mse(pred, gt)
        self.manual_backward(loss_g)
        opt.step()
        log_dict = {
            "lr": opt.param_groups[0]["lr"],
            "loss": loss_g.item(),
            "mse": F.mse_loss(pred.detach(), gt).item()
        }
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def lr_scheduler_step(self,scheduler, metric):
        scheduler.step(self.current_epoch)

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()

    def validation_step(self, batch, batch_idx):
        LR_t1, CSBS, LR_t2, HR_t1, gt = batch["LR_t1"], batch["CSBS"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        LR_t1 = batch["LR_t1"]
        out = self.forward(CSBS, LR_t2, HR_t1)
        pred = out["pred"]
        pred, gt = regularize_inputs(pred, gt)
        self.save_full_ref(pred, gt)
        if batch_idx in self.visual_idx:
            channel_indices = torch.tensor(self.rgb_c, device=self.device)
            LR_t1_rgb = torch.index_select(LR_t1, 1, channel_indices)
            LR_t2_rgb = torch.index_select(LR_t2, 1, channel_indices)
            HR_t1_rgb = torch.index_select(HR_t1, 1, channel_indices)
            CSBS_t1_rgb = torch.index_select(CSBS, 1, channel_indices)
            gt_rgb = torch.index_select(gt, 1, channel_indices)
            pred_rgb = torch.index_select(pred, 1, channel_indices)
            err_rgb = torch.abs(pred - gt).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            err_rgb /= torch.max(err_rgb)
            rgb_imgs = torch.cat([
                CSBS_t1_rgb,
                LR_t1_rgb,
                LR_t2_rgb,
                HR_t1_rgb,
                gt_rgb,
                pred_rgb,
                err_rgb], dim=0)

            if self.visual is None:
                self.visual = rgb_imgs
            else:
                self.visual = torch.cat([self.visual, rgb_imgs], dim=0)

    def on_validation_epoch_end(self):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.mean(self.metrics_all[metric])
            std = np.std(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(v, np.float64) and np.isnan(v) == False}
        self.log_dict(filtered_dict)
        filtered_dict["epoch"] = self.current_epoch
        csv_path = os.path.join(self.logger.save_dir, "metrics.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)

        grid = make_grid(self.visual, nrow=7, padding=2, normalize=False, scale_each=False, pad_value=0)
        image_grid = grid.permute(1, 2, 0).cpu().numpy()
        check_and_make(f"visual-{model_name}")
        save_path = f"visual-{model_name}/{self.current_epoch}.jpg"
        plt.imsave(save_path, image_grid)
        # self.logger.log_image(key="visual", images=[save_path])
        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        LR_t1, CSBS, LR_t2, HR_t1, gt = batch["LR_t1"], batch["CSBS"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        t_start = time.time()
        out = self.forward(CSBS, LR_t2, HR_t1)
        t_end = time.time()
        pred = out["pred"]
        len_patch = self.trainer.datamodule.dataset_test.LRs_arr.shape[1]
        pred, gt = regularize_inputs(pred, gt)
        self.record_metrics('Time', torch.tensor(t_end - t_start), "test")
        self.save_full_ref(pred, gt, "test")
        if (batch_idx+1) % len_patch == 0:
            self.save_metric((batch_idx+1) // len_patch)

    def on_test_epoch_start(self):
        self.reset_metrics("test")

    def save_metric(self, id_img=-1):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.mean(self.metrics_all[metric])
            std = np.std(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(
            v, np.float64) and np.isnan(v) == False and "std" not in k}
        print(filtered_dict)
        filtered_dict["epoch"] = f"testing-{id_img}"
        csv_path = os.path.join(self.logger.save_dir, f"test.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)
        self.reset_metrics("test")

    def save_full_ref(self, pred, gt, split="val"):
        data_range = (0., 1.)
        self.record_metrics('MAE', F.l1_loss(pred, gt), split)
        self.record_metrics('SSIM', MF.structural_similarity_index_measure(pred, gt, data_range=data_range), split)
        self.record_metrics('RMSE', MF.root_mean_squared_error_using_sliding_window(pred, gt), split)
        self.record_metrics('ERGAS', MF.error_relative_global_dimensionless_synthesis(pred, gt) / 16., split)
        self.record_metrics('SAM', MF.spectral_angle_mapper(pred, gt), split)
        self.record_metrics('PSNR', MF.peak_signal_noise_ratio(pred, gt, data_range=data_range), split)


    def reset_metrics(self, split="val"):
        self.eval_metrics = ['MAE', 'SAM', 'RMSE', 'ERGAS', 'PSNR', 'SSIM', "Time",]
        self.eval_metrics = [f"{split}/" + i for i in self.eval_metrics]
        tmp_results = {}
        for metric in self.eval_metrics:
            tmp_results.setdefault(metric, [])

        self.metrics_all = tmp_results
        self.visual = None

    def record_metrics(self, k, v, split="val"):
        if torch.isfinite(v):
            self.metrics_all[f'{split}/' + k].append(v.item())

