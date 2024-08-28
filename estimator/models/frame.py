import datetime
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchkeras import KerasModel, kerasmodel
from tqdm import tqdm


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(0.0)


class StepRunner(kerasmodel.StepRunner):
    """与kerasmodel.StepRunner区别在于本模块不在step中更新调度器的学习率"""
    def __call__(self, batch):
        features, labels = batch

        # loss
        with self.accelerator.autocast():  # type: ignore
            preds = self.net(features)
            loss = self.loss_fn(preds, labels)

        # backward()
        if self.stage == "train" and self.optimizer is not None:
            self.accelerator.backward(loss)  # type: ignore
            if self.accelerator.sync_gradients:  # type: ignore
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)  # type: ignore
            self.optimizer.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()  # type: ignore
        all_preds = self.accelerator.gather(preds)  # type: ignore
        all_labels = self.accelerator.gather(labels)  # type: ignore

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
                        for name, metric_fn in self.metrics_dict.items()}  # type: ignore

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


class EpochRunner(kerasmodel.EpochRunner):
    """与kerasmodel.EpochRunner区别在于本模块在epoch中进行调度器的学习率更新"""
    def __call__(self, dataloader):
        n = dataloader.size if hasattr(dataloader, 'size') else len(dataloader)
        loop = tqdm(enumerate(dataloader, start=1),
                    total=n,
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100
                    )
        epoch_losses = {}

        for step, batch in loop:
            with self.accelerator.accumulate(self.net):
                step_losses, step_metrics = self.steprunner(batch)
                step_log = dict(step_losses, **step_metrics)
                for k, v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

                if step < n:
                    loop.set_postfix(**step_log)

                    if hasattr(self, 'progress') and self.accelerator.is_local_main_process:
                        post_log = dict(**{'i': step, 'n': n}, **step_log)
                        self.progress.set_postfix(**post_log)  # type: ignore

                elif step == n:
                    # 更新调度器的学习率
                    if self.steprunner.lr_scheduler is not None:
                        self.steprunner.lr_scheduler.step()
                    epoch_metrics = step_metrics
                    epoch_metrics.update({self.stage + "_" + name: metric_fn.compute().item()
                                          for name, metric_fn in self.steprunner.metrics_dict.items()})
                    epoch_losses = {k: v / step for k, v in epoch_losses.items()}
                    epoch_log = dict(epoch_losses, **epoch_metrics)
                    loop.set_postfix(**epoch_log)

                    if hasattr(self, 'progress') and self.accelerator.is_local_main_process:
                        post_log = dict(**{'i': step, 'n': n}, **epoch_log)
                        self.progress.set_postfix(**post_log)  # type: ignore

                    for name, metric_fn in self.steprunner.metrics_dict.items():
                        metric_fn.reset()
                else:
                    break
        return epoch_log


class CurriculumModel(KerasModel):
    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint',
            patience=5, monitor="val_loss", mode="min", callbacks=None,
            plot=True, wandb=False, quiet=None,
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1):

        self.__dict__.update(locals())
        from accelerate import Accelerator
        from torchkeras.utils import colorful

        self.accelerator = Accelerator(mixed_precision=mixed_precision, cpu=cpu,
                                       gradient_accumulation_steps=gradient_accumulation_steps)

        device = str(self.accelerator.device)
        device_type = '🐌' if 'cpu' in device else ('⚡️' if 'cuda' in device else '🚀')
        self.accelerator.print(
            colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>"))

        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler)

        for key in self.kwargs:
            self.kwargs[key] = self.accelerator.prepare(self.kwargs[key])

        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
        # train_dataloader.size = train_data.size if hasattr(train_data, 'size') else len(train_data)
        # train_dataloader.size = min(train_dataloader.size, len(train_dataloader))

        # if val_data:
        #     val_dataloader.size = val_data.size if hasattr(val_data, 'size') else len(val_data)
        #     val_dataloader.size = min(val_dataloader.size, len(val_dataloader))

        self.history = {}
        callbacks = callbacks if callbacks is not None else []

        if bool(plot):
            from torchkeras.kerascallbacks import VisMetric, VisProgress
            callbacks = [VisMetric(), VisProgress()] + callbacks

        if wandb is not False:
            from torchkeras.kerascallbacks import WandbCallback
            project = wandb if isinstance(wandb, str) else 'torchkeras'
            callbacks.append(WandbCallback(project=project))

        self.callbacks = [self.accelerator.prepare(x) for x in callbacks]

        if self.accelerator.is_local_main_process:
            [cb.on_fit_start(model=self) for cb in self.callbacks if hasattr(cb, 'on_fit_start')]

        start_epoch = 1 if self.from_scratch else 0

        if bool(plot):
            quiet = True
        elif quiet is None:
            quiet = False

        quiet_fn = (lambda epoch: quiet) if isinstance(quiet, bool) else (
            (lambda epoch: epoch > quiet) if isinstance(quiet, int) else quiet)

        for epoch in range(start_epoch, epochs + 1):
            train_data.dataset.step(epoch)
            if val_data:
                val_data.dataset.step(epoch)
            # train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
            train_dataloader.size = train_data.size if hasattr(train_data, 'size') else len(train_data)
            train_dataloader.size = min(train_dataloader.size, len(train_dataloader))

            if val_data:
                val_dataloader.size = val_data.size if hasattr(val_data, 'size') else len(val_data)
                val_dataloader.size = min(val_dataloader.size, len(val_dataloader))
                should_quiet = quiet_fn(epoch)

            if not should_quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n" + "==========" * 8 + "%s" % nowtime)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs) + "\n")

            # 1，train -------------------------------------------------
            train_step_runner = self.StepRunner(
                net=self.net,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),
                optimizer=self.optimizer if epoch > 0 else None,
                lr_scheduler=self.lr_scheduler if epoch > 0 else None,
                **self.kwargs
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, should_quiet)
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                [cb.on_train_epoch_end(model=self) for cb in self.callbacks
                 if hasattr(cb, 'on_train_epoch_end')]

            # 2，validate -------------------------------------------------
            if val_dataloader is not None:
                val_step_runner = self.StepRunner(
                    net=self.net,
                    loss_fn=self.loss_fn,
                    accelerator=self.accelerator,
                    stage="val",
                    metrics_dict=deepcopy(self.metrics_dict),
                    **self.kwargs
                )
                val_epoch_runner = self.EpochRunner(val_step_runner, should_quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                [cb.on_validation_epoch_end(model=self) for cb in self.callbacks
                 if hasattr(cb, 'on_validation_epoch_end')]

            # self.save_ckpt(f'{ckpt_path}/latest.pt', accelerator=self.accelerator)
            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

            if best_score_idx == len(arr_scores) - 1 and self.accelerator.is_local_main_process:
                self.save_ckpt(f'{ckpt_path}/best.pt', accelerator=self.accelerator)
                if not should_quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor, arr_scores[best_score_idx])))

            if len(arr_scores) - best_score_idx > patience:
                break

        if self.accelerator.is_local_main_process:
            dfhistory = pd.DataFrame(self.history)
            [cb.on_fit_end(model=self) for cb in self.callbacks
             if hasattr(cb, 'on_fit_end')]
            if epoch < epochs:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>> \n"
                ).format(monitor, patience))
            self.net = self.accelerator.unwrap_model(self.net)
            self.net.cpu()
            self.load_ckpt(f'{ckpt_path}/best.pt')
            return dfhistory


CurriculumModel.StepRunner = StepRunner
CurriculumModel.EpochRunner = EpochRunner
