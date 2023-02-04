#!/usr/bin/env python
# encoding: utf-8
'''
@author: Shi Tong
@file: base_model.py
@time: 2022/11/9 14:12
'''
import torch
import os
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.criteria import MaskedMSELoss
from utils.vis_utils import save_depth_as_uint16png_upload, save_depth_as_uint8colored
from utils.metrics import AverageMeter
from utils.logger import logger
from typing import Dict, Any


class LightningBaseModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.depth_criterion = MaskedMSELoss()
        self.coarse_average_meter = AverageMeter()
        self.refined_average_meter = AverageMeter()
        self.test_average_meter = AverageMeter()
        self.mylogger = logger(args)

    def configure_optimizers(self):
        # *optimizer
        model_bone_params = [p for _, p in self.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_bone_params,
                                     lr=self.args['train_params']['learning_rate'],
                                     weight_decay=self.args['train_params']['weight_decay'],
                                     betas=(0.9, 0.99))

        # *lr_scheduler
        if self.args['train_params']['lr_scheduler'] == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args['train_params']['decay_step'], gamma=self.args['train_params']['decay_rate'])
        elif self.args['train_params']['lr_scheduler'] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer,
                                             mode='max',
                                             factor=self.args['train_params']['decay_rate'],
                                             patience=self.args['train_params']['decay_step'],
                                             verbose=True)
        elif self.args['train_params']['lr_scheduler'] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs - 4,
                eta_min=1e-5,
            )
        else:
            raise NotImplementedError('in base_model.py : lr_scheduler wrong, config lr_scheduler in .yaml')

        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def forward(self, data, cd=True) -> Any:
        pass

    def training_step(self, data: Dict, batch_idx):
        data = self.forward(data)
        coarse_output = data['coarse_output']
        refined_output = data['refined_output']
        gt = data['gt']
        coarse_loss = self.depth_criterion(coarse_output, gt)
        refined_loss = self.depth_criterion(refined_output, gt)
        if self.current_epoch < self.args['train_params']['train_stage0']:
            loss = 0.6 * coarse_loss + 0.4 * refined_loss
        elif self.current_epoch < self.args['train_params']['train_stage1']:
            loss = 0.9 * coarse_loss + 0.1 * refined_loss
        else:
            loss = refined_loss
        self.log('train/loss', loss.item())
        loss_dict = {'coarse_loss': coarse_loss.item(), 'refined_loss': refined_loss.item()}
        self.log('train/2_loss', loss_dict)
        return loss

    def validation_step(self, data, batch_idx):
        data = self.forward(data)
        coarse_output = data['coarse_output']
        refined_output = data['refined_output']
        gt = data['gt']

        self.coarse_average_meter.update(coarse_output, gt)
        self.refined_average_meter.update(refined_output, gt)

        rmse_dict = {'coarse_rmse': self.coarse_average_meter.rmse, 'refined_rmse': self.refined_average_meter.rmse}
        self.log('val/2_rmse', rmse_dict, on_step=True)

        self.mylogger.conditional_save_img_comparison(batch_idx, data, self.current_epoch, coarse_output, refined_output, gt)
        return

    def test_step(self, data, batch_idx):
        data = self.forward(data)
        if self.args.network_model == '_2dpaenet':
            pred = data['fuse_output']
        elif self.args.network_model == '_2dpapenet':
            pred = data['refined_depth']
        else:
            raise NotImplementedError('in base_model.py : test wrong, wrong network_model: ' + self.args.network_model)
        if self.args.mode == 'val':
            gt = data['gt']
            self.test_average_meter.update(pred, gt)
        else:
            str_i = str(self.current_epoch * self.args['train_params']['batch_size'] + batch_idx)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(self.args['model_params']['data_folder_save'], path_i)
            save_depth_as_uint16png_upload(pred, path)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(self.args['model_params']['data_folder_save'] + 'color/', path_i)
            save_depth_as_uint8colored(pred, path)

    def validation_epoch_end(self, outputs):
        self.coarse_average_meter.compute()
        self.refined_average_meter.compute()
        self.log('val/rmse', self.refined_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
        # rmse_dict = {'mid_rmse': self.mid_average_meter.sum_rmse, 'cd_rmse': self.cd_average_meter.sum_rmse}
        # self.log('val/mid_cd_rmse', rmse_dict, on_epoch=True, sync_dist=True)
        self.mylogger.conditional_save_info(self.coarse_average_meter, self.current_epoch)
        self.mylogger.conditional_save_info(self.refined_average_meter, self.current_epoch)
        if self.refined_average_meter.best_rmse > self.refined_average_meter.rmse:
            self.mylogger.save_img_comparison_as_best(self.current_epoch)
        self.coarse_average_meter.reset_all()
        self.refined_average_meter.reset_all()

    def test_epoch_end(self, outputs):
        if self.args.mode == 'val':
            self.test_average_meter.compute()
            self.mylogger.conditional_save_info(self.test_average_meter, self.current_epoch, False)

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print('detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
