################################################################################
#
#
#
################################################################################

import os
import torch

import pytorch_lightning as pl

import torchvision.transforms as transforms

from collections import OrderedDict
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN

from robotfashion.data import DeepFashion2

################################################################################

train_val_transform = transforms.Compose([transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])

num_classes = 14


################################################################################

# function to merge a list of inputs into a single batch
def collate(inputs):
    images = list()
    labels = list()

    for image, label in inputs:
        images.append(image)
        labels.append(label)

    return images, labels


def freeze_resnet50_fpn_backbone(model: FasterRCNN):
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad_(False)


class FasterRCNNWithRobotFashion(pl.LightningModule):
    def __init__(self, hparams):
        super(FasterRCNNWithRobotFashion, self).__init__()

        self.hparams = hparams

        self.num_data_loaders = hparams.num_data_loaders
        self.batch_size = hparams.batch_size
        self.data_folder_path = hparams.data_folder_path
        self.df2_password = hparams.df2_password
        self.subset_ratio = hparams.subset_ratio
        self.freeze_backbone = hparams.freeze_backbone

        self._faster_rcnn_model: FasterRCNN = fasterrcnn_resnet50_fpn(
            pretrained_backbone=True, num_classes=14
        )

        if self.freeze_backbone:
            freeze_resnet50_fpn_backbone(self._faster_rcnn_model)

        self._num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._prev_epoch = -1

    def forward(self, x):
        # input should be tuple (images: List[Tensor], labels: List[Dict[Tensor]]

        return self._faster_rcnn_model(*x)

    def training_step(self, batch, batch_idx):
        # a batch exists out of tuple (images: List[Tensor], labels: List[Dict[Tensor]]
        losses_dict = self.forward(batch)

        loss_classifier = losses_dict["loss_classifier"]
        loss_box_reg = losses_dict["loss_box_reg"]
        loss_objectness = losses_dict["loss_objectness"]
        loss_rpn_box_reg = losses_dict["loss_rpn_box_reg"]

        total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

        log_dict = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
            "total_loss": total_loss,
        }

        output = OrderedDict(
            {"loss": total_loss, "progress_bar": log_dict, "log": log_dict}
        )

        return output

    def validation_step(self, batch, batch_idx):
        # our validation should also be in training mode
        self.train()

        # a batch exists out of tuple (images: List[Tensor], labels: List[Dict[Tensor]]
        losses_dict = self.forward(batch)

        print(losses_dict)

        loss_classifier = losses_dict["loss_classifier"]
        loss_box_reg = losses_dict["loss_box_reg"]
        loss_objectness = losses_dict["loss_objectness"]
        loss_rpn_box_reg = losses_dict["loss_rpn_box_reg"]

        total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

        log_dict = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
            "val_loss": total_loss,
        }

        # lightning expects to be in eval mode
        self.eval()

        return OrderedDict(log_dict)

    def validation_end(self, outputs):
        """
        outputs -- list of outputs from each validation step
        """
        # The outputs here are strictly for progress bar
        loss_classifier = torch.stack([x["loss_classifier"] for x in outputs]).mean()
        loss_box_reg = torch.stack([x["loss_box_reg"] for x in outputs]).mean()
        loss_objectness = torch.stack([x["loss_objectness"] for x in outputs]).mean()
        loss_rpn_box_reg = torch.stack([x["loss_rpn_box_reg"] for x in outputs]).mean()
        total_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        log_dict = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
            "val_loss": total_loss,
        }

        output = OrderedDict({"progress_bar": log_dict, "log": log_dict})

        return output

    def test_step(self, batch, batch_nb):
        raise NotImplementedError()

    def test_end(self, outputs):
        raise NotImplementedError()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        train_data = DeepFashion2(
            self.data_folder_path,
            mode="train",
            download_if_missing=True,
            password=self.df2_password,
            transform=train_val_transform,
            subset_ratio=self.subset_ratio,
        )

        data_loader = DataLoader(
            train_data,
            num_workers=self.num_data_loaders,
            batch_size=self.batch_size,
            collate_fn=collate,
        )

        print("train len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        val_data = DeepFashion2(
            self.data_folder_path,
            mode="val",
            download_if_missing=True,
            password=self.df2_password,
            transform=train_val_transform,
            subset_ratio=self.subset_ratio,
        )

        data_loader = DataLoader(
            val_data,
            num_workers=self.num_data_loaders,
            batch_size=self.batch_size,
            collate_fn=collate,
        )

        print("validation length", len(data_loader))
        return data_loader

    # @pl.data_loader
    # def test_dataloader(self):
    #     raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyper-parameters of this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument("--num-data-loaders", default=4, type=int)
        parser.add_argument("--batch-size", default=4, type=int)
        parser.add_argument("--data-folder-path", default=os.getcwd(), type=str)
        parser.add_argument("--df2-password", default=None, type=str)
        parser.add_argument("--subset-ratio", default=1, type=float)
        parser.add_argument("--freeze-backbone", default=False, type=bool)

        return parser
