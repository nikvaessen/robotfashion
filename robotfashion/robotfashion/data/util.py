################################################################################
# Utility folder for collecting data
#
################################################################################

import os
import hashlib
import requests
import subprocess
import zipfile


################################################################################
# Download files from google drive
# Taken from https://stackoverflow.com/a/39225039


def download_file_from_google_drive(
    gd_id, destination, show_progress=False, data_length_hint=None
):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination, data_length_hint):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            dl = 0
            c = 0

            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    c += 1
                    dl += len(chunk)

                    if show_progress:
                        print(f"\rchunk {c}, total data: {dl}", end="")

                        if data_length_hint is not None:
                            print(
                                f" done: {round(dl / data_length_hint * 100, 4)}%",
                                end="",
                            )

                        print("", end="", flush=True)
            print()

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": gd_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": gd_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination, data_length_hint)


################################################################################
# data integrity check (both file and folder hashing)
# partially taken from https://stackoverflow.com/a/22058673


def sha256_hash_file(file_path):
    sha256 = hashlib.sha256()
    buf_size = 65536  # 64 kb chunks

    with open(file_path, "rb") as f:
        while True:
            data = f.read(buf_size)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def recurse_get_all_files_in_folder(folder_path):
    files = []
    subfolders = []

    for f in sorted(os.listdir(folder_path)):
        if os.path.isdir(f):
            subfolders += [f]

        files += [f]

    for sf in subfolders:
        files += recurse_get_all_files_in_folder(sf)

    return files


def sha256_hash_folder(folder_path):
    folder_files = recurse_get_all_files_in_folder(folder_path)

    str_to_hash = ""

    for f in folder_files:
        str_to_hash += f

    sha256 = hashlib.sha256()
    sha256.update(str_to_hash.encode())

    return sha256.hexdigest()


def check_folder_integrity(folder_path, original_hash):
    folder_hash = sha256_hash_folder(folder_path)

    return folder_hash == original_hash


def check_file_integrity(file_path, original_hash):
    file_hash = sha256_hash_file(file_path)

    return file_hash == original_hash


################################################################################
# unzip utility


def is_unzip_available():
    command = ["unzip", "--help"]

    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def unzip_command(zip_file_path, password=None):
    path, fn = os.path.split(zip_file_path)

    command = ["unzip", "-u"]

    if password is not None:
        command += ["-P", password]

    command += [fn]

    subprocess.check_call(command, cwd=path, stdout=subprocess.DEVNULL)


################################################################################
# Verifying and downloading files


def has_correct_folder_structure(root_path, folders, dataset_name):
    # root_path = get_root_data_folder(working_path)
    #
    # folders = [
    #     # order:
    #     # 1. folder name
    #     # 2. sha256 hash of all file and subfolder names
    #     #    concatenated to a string (without spaces as separation)
    #     (
    #         "validation",
    #         "a87d16eee207a902b5d3b5bb2ad9f92f0456ffd992b326e1f3a1dfbbc260d38e",
    #     ),
    #     (
    #         "json_for_test",
    #         "38f8e52f2a4d6e99b190d2ad71ecabdd397d9dc60673b303613ee16f99b0fdac",
    #     ),
    #     ("train", "a87d16eee207a902b5d3b5bb2ad9f92f0456ffd992b326e1f3a1dfbbc260d38e"),
    #     (
    #         "json_for_validation",
    #         "0868b572600747de8308160e4cf9eaaeeccf9a3ceab76e6e9bb1a29ba49e07db",
    #     ),
    #     ("test", "6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d"),
    # ]

    print(f"verifying {dataset_name} data is available on machine")

    for folder_name, hsh in folders:
        path = os.path.join(root_path, folder_name)

        if not os.path.isdir(path) or not check_folder_integrity(path, hsh):
            print(
                f"failed to find {dataset_name} data: could not find {folder_name} at {path}"
            )
            return False

    print(f"successfully found {dataset_name} data")
    return True


def maybe_download_and_unzip_data(root_path, download_links, password: str = None):
    if not is_unzip_available():
        print(
            "WARNING\n you have not installed the unzip command."
            + "Decrypting the zip files will take SEVERAL HOURS"
        )

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
                if password is not None:
                    zf.setpassword(password.encode())

                zf.extractall(root_path)
