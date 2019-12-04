import os

from robotfashion.data import RobotFashion


script_path, _ = os.path.split(os.path.abspath(__file__))


def main():
    print("Downloading Robotfashion to", script_path)
    RobotFashion(script_path, "train", download_if_missing=True)


if __name__ == "__main__":
    main()
