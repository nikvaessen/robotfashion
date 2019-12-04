import os
import json

import numpy as np
import torch as t

from .util import has_correct_folder_structure, maybe_download_and_unzip_data

from torchvision.datasets import VisionDataset
from PIL import Image


class RobotFashion(VisionDataset):
    train_mode = "train"
    val_mode = "val"
    test_mode = "test"
    modes = [train_mode, val_mode, test_mode]

    def __init__(
        self,
        working_path: str,
        mode: str,
        download_if_missing: bool = False,
        subset_ratio=1,
        transform=None,
    ):
        super().__init__(working_path, transform=transform, target_transform=None)

        super().__init__(working_path, transform=transform, target_transform=None)

        if not has_correct_folder_structure(
            self._get_root_data_folder(), self.get_folders(), self.get_dataset_name()
        ):
            if not download_if_missing:
                raise ValueError(
                    f"cannot find (valid) {self.get_dataset_name()} data."
                    + " Set download_if_missing=True to download dataset"
                )

            maybe_download_and_unzip_data(
                self._get_root_data_folder(), self.get_download_links()
            )

            if not has_correct_folder_structure(
                self._get_root_data_folder(),
                self.get_folders(),
                self.get_dataset_name(),
            ):
                raise Exception("Downloading and/or unzipping data failed")

        if mode not in RobotFashion.modes:
            raise ValueError(f"mode {mode} should be one of {RobotFashion.modes}")

        if subset_ratio <= 0 or subset_ratio > 1:
            raise ValueError(f"subset ratio {subset_ratio} needs to be in (0, 1]")
        else:
            self.subset_ratio = subset_ratio

        self.mode = mode

        if mode == RobotFashion.train_mode:
            self.image_paths, self.label_paths = self.load_train_data()
        elif mode == RobotFashion.val_mode:
            self.image_paths, self.label_paths = self.load_val_data()
        else:
            self.image_paths, self.label_paths = self.load_test_data()

    def _get_root_data_folder(self):
        return os.path.join(self.root, self.get_data_folder_name())

    def load_train_data(self):
        return self.load_data(os.path.join(self._get_root_data_folder(), "train"))

    def load_val_data(self):
        return self.load_data(os.path.join(self._get_root_data_folder(), "validation"))

    def load_test_data(self):
        # return self.load_data(
        #     os.path.join(get_root_data_folder(self.root), "test")
        # )
        raise NotImplementedError("labels of test data are not published")

    def __getitem__(self, index):
        image = self.load_image(self.image_paths[index])
        label = self.load_label(self.label_paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        n = len(self.image_paths)

        return int(self.subset_ratio * n)

    @staticmethod
    def load_data(data_dir):
        image_dir = os.path.join(data_dir, "image")
        annos_dir = os.path.join(data_dir, "xml")

        image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if os.path.isfile(os.path.join(image_dir, f))
        ]
        label_paths = [
            os.path.join(annos_dir, f)
            for f in sorted(os.listdir(annos_dir))
            if os.path.isfile(os.path.join(annos_dir, f))
        ]

        if len(image_paths) != len(label_paths):
            raise ValueError("length of images and labels doesn't match")

        return image_paths, label_paths

    @staticmethod
    def load_image(image_path):
        img = Image.open(image_path)

        return img

    @staticmethod
    def load_label(label_path):
        # During training, the model expects both the input tensors, as well as a targets (list of dictionary),
        # containing:
        #     - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
        #       between 0 and H and 0 and W
        #     - labels (Int64Tensor[N]): the class label for each ground-truth box

        with open(label_path, "r") as f:
            obj = json.load(f)

        items = []
        count = 0
        while True:
            count += 1
            key = f"item{count}"

            if key in obj:
                items.append(obj[key])
            else:
                break

        n = len(items)
        boxes = np.zeros((n, 4))
        labels = np.zeros((n,))

        for idx, item in enumerate(items):
            boxes[idx, :] = item["bounding_box"]
            labels[idx] = item["category_id"]

        return {"boxes": t.tensor(boxes).float(), "labels": t.tensor(labels).long()}

    @classmethod
    def get_data_folder_name(cls):
        return f"{cls.get_dataset_name()}_data_folder"

    @staticmethod
    def get_dataset_name():
        return "robotfashion"

    @staticmethod
    def get_download_links():
        return [
            # order:
            # 1. google drive id,
            # 2. file name,
            # 3. sha256 hash of zipfile,
            # 4. data length of zipfile
            ("-UE-LX8sks8eAcGLL-9QDNyNt6VgP", "test.zip", "", 0),
            ("", "train.zip", "", 0),
            ("", "validation.zip", "", 0),
        ]

    @staticmethod
    def get_folders():
        return [
            # order:
            # 1. folder name
            # 2. sha256 hash of all file and subfolder names
            #    concatenated to a string (without spaces as separation)
            ("validation", ""),
            ("train", ""),
            ("test", ""),
        ]
