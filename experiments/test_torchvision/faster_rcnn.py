from torchvision.models.detection import fasterrcnn_resnet50_fpn


def main():
    model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=2)


if __name__ == "__main__":
    main()
