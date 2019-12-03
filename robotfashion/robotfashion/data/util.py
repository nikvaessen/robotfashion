################################################################################
# Utility folder for collecting data
#
################################################################################

import os
import hashlib
import requests
import subprocess

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
# File integrity check
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


def is_unzip_available():
    command = ["unzip", "--help"]

    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def unzip_command(zip_file_path, password):
    path, fn = os.path.split(zip_file_path)
    command = ["unzip", "-u", "-P", password, fn]

    subprocess.check_call(command, cwd=path, stdout=subprocess.DEVNULL)
