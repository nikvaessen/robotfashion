import os

from robotfashion.data import DeepFashion2
from robotfashion.data.util import sha256_hash_folder

root_path = (
    "/home/nik/kth/y2/project_in_ds/realsense-cli/experiments/deepfashion_2_data_folder"
)
#
# for path in os.listdir(root_path):
#     p = os.path.join(root_path, path)
#
#     if os.path.isdir(p):
#         print(path, sha256_hash_folder(p))

DeepFashion2("2019Deepfashion2**")
