# %% import necessary libraries
import os
from argparse import Namespace

import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader
from torchkeras import kerascallbacks
from torchkeras.kerascallbacks import WandbCallback

import Datasets
import feature_modules
import models
import wandb
from frame import CurriculumModel

# %% configuration
c = 1500
fc = 42500
fs_factor = 8
fs = fs_factor * fc
d = 0.5
K = 1.0
r_range = (100, 2500)
sample_interval = 30

dataset_path = f'/root/autodl-tmp/dataset/fc-{fc}_fs_factor-{fs_factor}_d-{d}_K-{K}_r-{r_range[0]}-{r_range[1]}_i-{sample_interval}'
train_path = f'{dataset_path}/train'
val_path = f'{dataset_path}/val'

# see: https://docs.wandb.ai/guides/artifacts/storage
os.environ['WANDB_DIR'] = '/root/autodl-tmp/'
os.environ['WANDB_CACHE_DIR'] = '/root/autodl-tmp/cache/'

features = {
    feature_modules.STFT_Magnitude_Feature.__name__: feature_modules.STFT_Magnitude_Feature,
    feature_modules.CPSD_Phase_Feature.__name__: feature_modules.CPSD_Phase_Feature,
    feature_modules.CPSD_Phase_Diff_Feature.__name__: feature_modules.CPSD_Phase_Diff_Feature,
}

project = 'AC-LSTM'
sweep = True
config = Namespace(
    label_type='direction',
    curriculum=(-1, -1, -1, -1),
    step_size=10,
    gamma=0.5,
    feature='CPSD_Phase_Diff_Feature',
    batch_size=90,
    lr=1e-6,
)
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_L1',
        'goal': 'minimize',
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2,
        's': 3,
    },
    'parameters': {
        'label_type': {'value': config.label_type},
        'curriculum': {'value': config.curriculum},  # 同时在ideal, 30, 20dB SNR上训练
        'step_size': {'value': config.step_size},
        'gamma': {'value': config.gamma},
        'batch_size': {'value': 2},
        'feature': {
            'values': list(features.keys())
        },
        'alpha': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1,
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2,
        },
    },
}


# %% train script definition
def build_model(config):
    checkpoint = torch.load(f'models/{config.feature}_ResNet18.pt')
    backbone_wrapper = models.Custom_ResNet18(features[config.feature], fs, fc, 20e3, 60e3, config.label_type)
    backbone_wrapper.load_state_dict(checkpoint)
    net = models.Resnet18_attConvLSTM(backbone_wrapper, fs, fc, 20e3, 60e3, config.label_type)
    return net


def train(config, wandb_cb):
    exp_path = f'./exp/{config.feature}-AC-LSTM'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    ds_train = Datasets.Curriculum_Array_Dataset(train_path, seq=True, curriculum=config.curriculum, label_type=config.label_type, distance_range=r_range)
    ds_val = Datasets.Curriculum_Array_Dataset(val_path, seq=True, curriculum=config.curriculum, label_type=config.label_type, distance_range=r_range)
    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=16, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True, num_workers=16, drop_last=False)

    net = build_model(config)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    def loss_func_factory(alpha):
        def loss_func(y_pred, y_true):
            y_pred, ag, a1, a2, a3, a4 = y_pred
            BCELoss = nn.BCELoss()
            ag1 = nn.Upsample(size=(241, 4), mode='bilinear', align_corners=True)(ag)
            ag2 = nn.Upsample(size=(121, 2), mode='bilinear', align_corners=True)(ag)
            ag3 = nn.Upsample(size=(61, 1), mode='bilinear', align_corners=True)(ag)
            ag4 = nn.Upsample(size=(31, 1), mode='bilinear', align_corners=True)(ag)
            la1 = BCELoss(a1, ag1)
            la2 = BCELoss(a2, ag2)
            la3 = BCELoss(a3, ag3)
            la4 = BCELoss(a4, ag4)
            return nn.L1Loss()(y_pred, y_true) + alpha * (la1 + la2 + la3 + la4)
        return loss_func

    class Label_Only_MeanAbsoluteError(torchmetrics.MeanAbsoluteError):
        def update(self, pred, target):
            return super().update(pred[0], target)

    model = CurriculumModel(
        net,
        loss_fn=loss_func_factory(config.alpha),
        metrics_dict={
            'L1': Label_Only_MeanAbsoluteError(),
        },
        optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
    )

    model.fit(
        dl_train,
        dl_val,
        epochs=10,
        ckpt_path=exp_path,
        patience=3,
        monitor='val_loss',
        mode='min',
        plot=False,
        quiet=True,
        callbacks=[
            kerascallbacks.VisProgress(),
            kerascallbacks.VisMetric(save_path=f'{exp_path}/history.jpg'),
            wandb_cb,
        ]
    )


# %% train
if __name__ == '__main__':
    if sweep:
        sweep_id = wandb.sweep(sweep_config, project=project)

    def sweep_callback():
        wandb_cb = WandbCallback(project=project, config=config)
        with wandb.init(name=wandb_cb.name):
            train(wandb.config, wandb_cb)

    if sweep:
        wandb.agent(sweep_id, sweep_callback, count=50)
    else:
        train(config, wandb_cb=None)
