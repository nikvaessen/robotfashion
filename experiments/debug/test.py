import os

from robotfashion.data import DeepFashion2, RobotFashion
from robotfashion.data.util import sha256_hash_folder

from robotfashion.models.faster_rcnn.model import FasterRCNNWithRobotFashion
from robotfashion.models.faster_rcnn.trainer import get_parser

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

import torchsummary


def calculate_folder_hashes():
    root_path = "/home/nik/kth/y2/project_in_ds/realsense-cli/experiments/robotfashion_data_folder"

    for path in os.listdir(root_path):
        p = os.path.join(root_path, path)

        if os.path.isdir(p):
            print(path, sha256_hash_folder(p))


def get_data_loader_dist(data_loader):
    dist = {}
    for batch in data_loader:
        _, label = batch
        classes = label[0]["labels"]

        for cls in classes:
            cls = cls.int().item()

            if cls in dist:
                dist[cls] += 1
            else:
                dist[cls] = 0

    print(dist)


def main():
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

    rf = RobotFashion(
        os.getcwd(),
        "train",
        download_if_missing=True,
        transform=train_val_transform,
        subset_ratio=1,
    )

    print("samples in data:", len(rf))

    def collate(inputs):
        images = list()
        labels = list()

        for image, label in inputs:
            images.append(image)
            labels.append(label)

        return images, labels

    data_loader = DataLoader(rf, num_workers=1, batch_size=1, collate_fn=collate)

    # get_data_loader_dist(data_loader)

    x = data_loader.__iter__().__next__()
    print(x)

    hparams = get_parser()
    model: FasterRCNNWithRobotFashion = FasterRCNNWithRobotFashion(hparams)

    y = model(x)
    print(y)

    #
    # for p in model.named_parameters():
    #     a, b = p
    #     print(a, b.shape, f"frozen?:{not b.requires_grad}")


if __name__ == "__main__":
    main()
