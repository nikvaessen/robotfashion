import os

from robotfashion.data import DeepFashion2
from robotfashion.data.util import sha256_hash_folder

from robotfashion.models.faster_rcnn.model import FasterRCNNWithRobotFashion

from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms


def calculate_folder_hashes():
    root_path = (
        "/home/nik/kth/y2/project_in_ds/realsense-cli/experiments/deepfashion_2_data_folder"
    )

    for path in os.listdir(root_path):
        p = os.path.join(root_path, path)

        if os.path.isdir(p):
            print(path, sha256_hash_folder(p))


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
    df2 = DeepFashion2(os.getcwd(), "train", download_if_missing=True, password="2019Deepfashion2**",
                       transform=train_val_transform)

    def collate(inputs):
        images = list()
        labels = list()

        for image, label in inputs:
            images.append(image)
            labels.append(label)

        return images, labels

    data_loader = DataLoader(
        df2,
        num_workers=1,
        batch_size=2,
        collate_fn=collate
    )

    x = data_loader.__iter__().__next__()
    print(x)

    hparams = {"split": 0.9, "num_data_loaders": 4}
    model: FasterRCNNWithRobotFashion = FasterRCNNWithRobotFashion(hparams)
    y = model.forward(x)

    print(y)


if __name__ == '__main__':
    main()
