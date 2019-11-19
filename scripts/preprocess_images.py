#! /usr/bin/env python3

import os

from PIL import Image


def process_cloth_front_view_image(img_path):
    img = Image.open(img_path)

    img = img.crop((438, 0, 843, 720))

    return img


def process_cloth_back_view_image(img_path):
    img = Image.open(img_path)

    img = img.rotate(-90, expand=True)
    img = img.resize((405, 720))

    return img


def rotate_images_in_folder(folder_path):
    child_directories = []

    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)

        if os.path.isdir(path):
            child_directories.append(path)

        if ".png" in path:
            if "dev_829212071075" in path:
                img = process_cloth_back_view_image(path)
            else:
                img = process_cloth_front_view_image(path)
                pass

            img.save(path)
            print("processed", path)

    for c in child_directories:
        print("checking folder ", c)
        rotate_images_in_folder(c)


def main():
    cwd = os.getcwd()

    rotate_images_in_folder(cwd)


if __name__ == "__main__":
    main()
