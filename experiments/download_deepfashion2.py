import os

from robotfashion.data import DeepFashion2


script_path, _ = os.path.split(os.path.abspath(__file__))


def main():
    print("Downloading deepfashion2 to", script_path)
    password = input("What's the Deepfashion2 password?")
    DeepFashion2(script_path, "train", password=password, download_if_missing=True)


if __name__ == "__main__":
    main()
