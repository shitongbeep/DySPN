import numpy as np
from torchmetrics import Metric
import math
import torch

lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10


class AverageMeter(Metric):

    def __init__(self):
        super(AverageMeter, self).__init__(dist_sync_on_step=False, compute_on_step=True)
        self.count = 0.0
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.photometric = 0
        self.silog = 0

        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_photometric = 0
        self.sum_silog = 0
        self.reset_all()
        self.best_rmse = np.inf

    def reset_all(self):
        self.count = 0.0
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.photometric = 0
        self.silog = 0

        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_photometric = 0
        self.sum_silog = 0

    def update(self, output, target, photometric=0):
        valid_mask = target > 0.1

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.count = self.count+1
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm)**2).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask])**(-1)
        inv_target_km = (1e-3 * target[valid_mask])**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        self.photometric = float(photometric)

        self.sum_irmse += self.irmse
        self.sum_imae += self.imae
        self.sum_mse += self.mse
        self.sum_rmse += self.rmse
        self.sum_mae += self.mae
        self.sum_absrel += self.absrel
        self.sum_squared_rel += self.squared_rel
        self.sum_lg10 += self.lg10
        self.sum_delta1 += self.delta1
        self.sum_delta2 += self.delta2
        self.sum_delta3 += self.delta3
        self.sum_photometric += self.photometric
        self.sum_silog += self.silog

    def compute(self):
        if self.count == 0:
            return None
        self.sum_irmse /= self.count
        self.sum_imae /= self.count
        self.sum_mse /= self.count
        self.sum_rmse /= self.count
        self.sum_mae /= self.count
        self.sum_absrel /= self.count
        self.sum_squared_rel /= self.count
        self.sum_lg10 /= self.count
        self.sum_delta1 /= self.count
        self.sum_delta2 /= self.count
        self.sum_delta3 /= self.count
        self.sum_photometric /= self.count
        self.sum_silog /= self.count
        if self.best_rmse < self.sum_rmse:
            self.best_rmse = self.sum_rmse
