################################################################################
#
#
#
################################################################################

import os
import torch

import numpy as np
import pytorch_lightning as pl

import torchvision.transforms as transforms

from collections import OrderedDict
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.datasets import CIFAR10

################################################################################

train_val_transform = transforms.Compose(
    [
        # transforms.Pad((4, 4, 4, 4)),
        # transforms.RandomCrop((32, 32)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # values are between [0, 1], we want [-1, 1]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # values are between [0, 1], we want [-1, 1]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

num_classes = 14

################################################################################


class FasterRCNNWithRoboFashion(pl.LightningModule):
    def __init__(self, hparams):
        super(FasterRCNNWithRoboFashion, self).__init__()

        self.hparams = hparams

        self.split = hparams.train_val_split
        self.num_data_loaders = hparams.num_data_loaders

        self._fast_rcnn_model: FasterRCNN = fasterrcnn_resnet50_fpn(
            pretrained_backbone=True, num_classes=14
        )

        self._num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._prev_epoch = -1

    def forward(self, x):
        return self._fast_rcnn_model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)

        # Training metrics for monitoring
        labels_hat = torch.argmax(y_hat, dim=1)
        train_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        train_loss = F.cross_entropy(y_hat, y)
        logger_logs = {"train_acc": train_acc, "train_loss": train_loss}

        # loss is strictly required
        output = OrderedDict(
            {
                "loss": train_loss,
                "progress_bar": {"train_acc": train_acc},
                "log": logger_logs,
            }
        )

        return output

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        # validation metrics for monitoring
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_loss = F.cross_entropy(y_hat, y)

        return OrderedDict(
            {"val_loss": val_loss.clone().detach(), "val_acc": torch.tensor(val_acc)}
        )

    def validation_end(self, outputs):
        """
        outputs -- list of outputs ftom each validation step
        """
        # The outputs here are strictly for progress bar
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        logger_logs = {"val_acc": avg_acc, "val_loss": avg_loss}

        output = OrderedDict({"progress_bar": logger_logs, "log": logger_logs})

        return output

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        # validation metrics for monitoring
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_loss = F.cross_entropy(y_hat, y)

        return OrderedDict(
            {
                "test_loss": test_loss.clone().detach(),
                "test_acc": torch.tensor(test_acc),
            }
        )

    def test_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        return OrderedDict(
            {
                "progress_bar": {"loss": avg_loss, "accuracy": avg_acc},
                "log": {"test_acc": avg_acc, "test_loss": avg_loss},
            }
        )

    def configure_optimizers(self):
        raise NotImplemented()

    def get_train_val_sampler(self, num_samples):
        indices = list(range(num_samples))

        if self.split == 1:
            train_idx = indices
            split_idx_val = int(np.floor(0.1 * num_samples))
            val_idx = indices[:split_idx_val]
        else:
            split_idx = int(np.floor(self.split * num_samples))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        return train_sampler, val_sampler

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        train_data = CIFAR10(
            os.getcwd(), train=True, download=True, transform=train_val_transform
        )

        train_sampler, _ = self.get_train_val_sampler(len(train_data))

        data_loader = DataLoader(
            train_data,
            num_workers=self.num_data_loaders,
            batch_size=self.bs,
            sampler=train_sampler,
        )

        print("train len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        val_data = CIFAR10(
            os.getcwd(), train=True, download=True, transform=train_val_transform
        )

        _, val_sampler = self.get_train_val_sampler(len(val_data))

        data_loader = DataLoader(
            val_data,
            num_workers=self.num_data_loaders,
            batch_size=self.bs,
            sampler=val_sampler,
        )

        print("val len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            CIFAR10(os.getcwd(), train=False, download=True, transform=test_transform),
            batch_size=self.hparams.batch_size,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--adaptivity-rate", default=10 ** -4, type=float)
        parser.add_argument("--threshold", default=10 ** -8, type=float)
        parser.add_argument("--batch_size", default=50, type=int)
        parser.add_argument("--adam-lr", default=0.01, type=float)
        parser.add_argument("--decay-n-epochs", default=100, type=int)
        parser.add_argument("--decay-exponential", default=0.1, type=float)
        parser.add_argument("--train-val-split", default=0.90, type=float)

        return parser
