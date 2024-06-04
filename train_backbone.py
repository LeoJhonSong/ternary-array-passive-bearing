# %% import necessary libraries
import os
from argparse import Namespace

import torchmetrics
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torchkeras import kerascallbacks
from torchkeras.kerascallbacks import WandbCallback

import Datasets
import feature_modules
import models
from frame import CurriculumModel

# %% configuration
c = 1500
fc = 42500
fs_factor = 8
fs = fs_factor * fc
d = 0.5
K = 1.0
r_range = (100, 2500)
sample_interval = 1

dataset_path = f'/root/autodl-tmp/dataset/fc-{fc}_fs_factor-{fs_factor}_d-{d}_K-{K}_r-{r_range[0]}-{r_range[1]}_i-{sample_interval}'
train_path = f'{dataset_path}/train'
val_path = f'{dataset_path}/val'

features = {
    feature_modules.STFT_Magnitude_Feature.__name__: feature_modules.STFT_Magnitude_Feature,
    feature_modules.CPSD_Phase_Feature.__name__: feature_modules.CPSD_Phase_Feature,
    feature_modules.CPSD_Phase_Diff_Feature.__name__: feature_modules.CPSD_Phase_Diff_Feature,
}

project = 'backbone'

sweep = True
config = Namespace(
    label_type='direction',
    curriculum=(-1, -1, 300, 40),
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
        'min_iter': 5,
        'eta': 2,
        's': 2,
    },
    'parameters': {
        'label_type': {'value': config.label_type},
        'curriculum': {'value': config.curriculum},  # 同时在ideal, 30, 20dB SNR上训练
        'step_size': {'value': config.step_size},
        'gamma': {'value': config.gamma},
        'feature': {
            'values': list(features.keys())
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 24,
            'max': 90,
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3,
        },
    },
}
if sweep:
    sweep_id = wandb.sweep(sweep_config, project=project)


# %% train script definition
def train(config, wandb_cb):
    exp_path = f'./exp/{config.feature}'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    ds_train = Datasets.Curriculum_Array_Dataset(train_path, seq=False, curriculum=config.curriculum, label_type=config.label_type, distance_range=r_range)
    ds_val = Datasets.Curriculum_Array_Dataset(val_path, seq=False, curriculum=config.curriculum, label_type=config.label_type, distance_range=r_range)
    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=16, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=True, num_workers=16, drop_last=False)

    net = models.Custom_ResNet18(features[config.feature], fs, fc, 20e3, 60e3, config.label_type)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    model = CurriculumModel(
        net,
        loss_fn=nn.L1Loss(),
        metrics_dict={
            'L1': torchmetrics.MeanAbsoluteError(),
        },
        optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
    )

    model.fit(
        dl_train,
        dl_val,
        epochs=50,
        ckpt_path=exp_path,
        patience=5,
        monitor='val_L1',
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
def sweep_callback():
    wandb_cb = WandbCallback(project=project, config=config)
    with wandb.init(name=wandb_cb.name):
        train(wandb.config, wandb_cb)


if sweep:
    wandb.agent(sweep_id, sweep_callback, count=50)
else:
    train(config, wandb_cb=None)
