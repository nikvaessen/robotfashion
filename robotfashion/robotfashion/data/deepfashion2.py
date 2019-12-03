import os
import zipfile

from .util import (
    download_file_from_google_drive,
    check_file_integrity,
    check_folder_integrity,
    is_unzip_available,
    unzip_command,
)

from torchvision.datasets import VisionDataset


def has_correct_folder_structure(working_path):
    root_path = os.path.join(working_path, "deepfashion_2_data_folder")

    folders = [
        # order:
        # 1. folder name
        # 2. sha256 hash of all file and subfolder names
        #    concatenated to a string (without spaces as separation)
        (
            "validation",
            "a87d16eee207a902b5d3b5bb2ad9f92f0456ffd992b326e1f3a1dfbbc260d38e",
        ),
        (
            "json_for_test",
            "38f8e52f2a4d6e99b190d2ad71ecabdd397d9dc60673b303613ee16f99b0fdac",
        ),
        ("train", "a87d16eee207a902b5d3b5bb2ad9f92f0456ffd992b326e1f3a1dfbbc260d38e"),
        (
            "json_for_validation",
            "0868b572600747de8308160e4cf9eaaeeccf9a3ceab76e6e9bb1a29ba49e07db",
        ),
        ("test", "6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d"),
    ]

    print("verifying DeepFashion2 data is available on machine")
    for folder_name, hsh in folders:
        path = os.path.join(root_path, folder_name)

        if not os.path.isdir(path) or not check_folder_integrity(path, hsh):
            print(
                f"failed to find DeepFashion2 data: could not find {folder_name} at {path}"
            )
            return False

    print("successfully found DeepFashion2 data")
    return True


def maybe_download_and_unzip_data(working_path, password: str):
    if not is_unzip_available():
        print(
            "WARNING\n you have not installed the unzip command."
            + "Decrypting the zip files will take SEVERAL HOURS"
        )

    root_path = os.path.join(working_path, "deepfashion_2_data_folder")

    download_links = [
        # order:
        # 1. google drive id,
        # 2. file name,
        # 3. sha256 hash of zipfile,
        # 4. data length of zipfile
        (
            "12DmrxXNtl0U9hnN1bzue4XX7nw1fSMZ5",
            "json_for_validation.zip",
            "1899b133c15b961c317cf03f589cdc8423fe16b290e534b642accad538656ab4",
            14895000,
        ),
        (
            "1hsa-UE-LX8sks8eAcGLL-9QDNyNt6VgP",
            "test.zip",
            "1a85367dc9c75fbac8645e397b93af11c86bc059ab718c1eee31b559b5b4598b",
            3341995077,
        ),
        (
            "1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK",
            "train.zip",
            "ec6f5d83f896f3abbb46bcfb9fdd6b9f544c0585344f862c214f6de899c495c7",
            10633411064,
        ),
        (
            "1O45YqhREBOoLudjA06HcTehcEebR0o9y",
            "validation.zip",
            "edabbdb57fae4b5039ff06e436cc0dfa15326424244bfac938e4a4d6f8db0259",
            1816223824,
        ),
    ]

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for gd_id, fn, hsh, length in download_links:
        save_path = os.path.join(root_path, fn)

        if os.path.exists(save_path):
            if check_file_integrity(save_path, hsh):
                print(f"found {fn} with correct hash. Skipping download...")
                continue
            else:
                print(f"found {fn} but is corrupted/incomplete. Downloading again...")
        else:
            print(f"downloading {fn}")

        download_file_from_google_drive(
            gd_id, save_path, show_progress=True, data_length_hint=length
        )

    for gd_id, fn, hsh, length in download_links:
        save_path = os.path.join(root_path, fn)

        print(f"extracting {fn} in {root_path}")

        if is_unzip_available():
            unzip_command(save_path, password)
        else:
            print("Did not find the unzip command. Unzipping will take several hours!")
            with zipfile.ZipFile(save_path) as zf:
                zf.setpassword(password.encode())
                zf.extractall(root_path)


class DeepFashion2(VisionDataset):
    train_mode = "train"
    val_mode = "val"
    test_mode = "test"
    modes = [train_mode, val_mode, test_mode]

    def __init__(
        self,
        working_path: str,
        password: str,
        mode: str,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            working_path, transform=transform, target_transform=target_transform
        )

        if not password:
            raise PermissionError(
                "Cannot access deepfashion2 data without the password."
                + " See https://github.com/switchablenorms/DeepFashion2#download-the-data"
            )

        if not has_correct_folder_structure(working_path):
            maybe_download_and_unzip_data(working_path, password)

            if not has_correct_folder_structure(working_path):
                raise Exception("Unable to get Deepfashion2 data")

        if mode not in DeepFashion2.modes:
            raise ValueError(f"mode {mode} should be one of {DeepFashion2.modes}")

        self.mode = mode



    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    DeepFashion2()
