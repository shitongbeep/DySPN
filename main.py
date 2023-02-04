import argparse
import importlib
import os

import pytorch_lightning as pl
import tensorboardX
import yaml
from dataloaders import kitti_loader
from easydict import EasyDict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler
from torch.utils.data.dataloader import DataLoader


def getargs():
    parser = argparse.ArgumentParser(description="Sparse to Dense")
    parser.add_argument('-c', '--configs', default="./config/2DPAPENet-kitti.yaml")
    configs = parser.parse_args()
    with open(configs.configs, 'r') as config:
        args = yaml.safe_load(config)
    args.update(vars(configs))
    args = EasyDict(args)
    print(args)
    return args


def build_loader(args):
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None
    if args.model_params.mode == 'train':
        train_dataset = kitti_loader.KittiDepth('train', args)
        train_dataset_loader = DataLoader(train_dataset,
                                          batch_size=args.train_params.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=args.train_params.workers)
        val_dataset = kitti_loader.KittiDepth('val', args)
        val_dataset_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.train_params.workers)
    elif args.model_params.mode == 'val':
        val_dataset = kitti_loader.KittiDepth('val', args)
        val_dataset_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.train_params.workers)
    elif args.model_params.mode == 'test':
        test_dataset = kitti_loader.KittiDepth('test', args)
        test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    else:
        raise AttributeError('wrong mode')
    return train_dataset_loader, val_dataset_loader, test_dataset_loader


def get_model(args):
    model_file = importlib.import_module('network.' + args.model_params.network_model)
    return model_file.get_model(args)


if __name__ == "__main__":
    args = getargs()  # 加载参数
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args['train_params']['gpu']))
    num_gpu = len(args['train_params']['gpu'])
    tb_logger = pl_loggers.TensorBoardLogger(args['model_params']['log_folder'], name='kitti_depth_completion', default_hp_metric=False)
    profiler = SimpleProfiler(filename='profiler')
    if not os.path.exists(args['model_params']['mylog_folder']):
        os.makedirs(args['model_params']['mylog_folder'])
    pl.seed_everything(args['train_params']['seed'])

    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(args)
    my_model = get_model(args)
    # 保存和加载模型参数
    ckpt_path = None
    if os.path.isfile(args['model_params']['checkpoint'] + '.ckpt'):
        print('load pre-trained model...')
        my_model = my_model.load_from_checkpoint(args['model_params']['checkpoint'] + '.ckpt', args=args, strict=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath='.',
        filename=args['model_params']['checkpoint'],
        monitor=args['train_params']['monitor'],
        mode='min',
        save_last=True,
    )
    # 开始训练
    if args['model_params']['mode'] == 'train':
        print('Start training...')
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[i for i in range(num_gpu)],
            strategy='ddp',
            max_epochs=args['train_params']['epochs'],
            limit_train_batches=0.6,
            # limit_val_batches=10,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step'),
                EarlyStopping(monitor=args['train_params']['monitor'], patience=args['train_params']['stop_patience'], mode='min', verbose=True),
            ],
            logger=tb_logger,
            profiler=profiler,
            log_every_n_steps=args['train_params']['log_freq'],
            gradient_clip_algorithm='norm',
            gradient_clip_val=1
        )
        trainer.fit(my_model, train_dataset_loader, val_dataset_loader)
    else:
        if not os.path.exists(args['model_params']['data_folder_save']):
            os.mkdir(args['model_params']['data_folder_save'])
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        trainer = pl.Trainer(accelerator='gpu', devices=[i for i in range(num_gpu)], strategy='ddp', logger=tb_logger, profiler=profiler)
        if args['model_params']['mode'] == 'val':
            trainer.test(my_model, val_dataset_loader)
        elif args['model_params']['mode'] == 'test':
            trainer.test(my_model, test_dataset_loader)
