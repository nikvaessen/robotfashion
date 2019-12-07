import os
import numpy as np
import cv2
import re


def convert_depth_images(folder_path):
    count = 0
    slash = '\\'
    hyphen = '_'
    fold_list = ['pants', 'extra_outliers', 'session08novtshirtsparta','session08novtshirtspartb', 'session08novtshirtspartc',
            'session11nov_long_sleeve_outwear', 'session11nov_long_sleeve_pullover', 'session11novlong', 'session11novlong_2',
            'session11novshort', r'session12nov\long_sleeve_dress', r'session12nov\short_sleeve_dress', r'session12nov\skirts',
            r'session12nov\sling', r'session12nov\ufo',
            'session25oktbtrousers', r'session25oktshorts', 'session25okt',
            r'session12nov\vest_dress', r'session12nov\vests']
    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)

        if os.path.isdir(path):
            child_directories.append(path)

        if "color.png" in path:
            file = path.split('\\')[-1]
            file_split = re.split("_(\d{10}\w+)color", file)

            depth_npz = file_split[1] + 'depth.npz'
            color_npz = file_split[1] + 'color.npz'
            save_as = file_split[0] + '_' + file_split[1] + 'depth.png'
            print(hyphen.join(file_split[0].split('_')[:-1]))
            for fold in fold_list:
                folder_selected = slash.join(['D:\clothing-data',fold])
                color_npz_path = os.path.join(os.path.join(folder_selected, file_split[0]), color_npz)
                if os.path.exists(color_npz_path):
                    print("Exists"+str(color_npz_path))
                    break
                else:
                    print("Not found in " + str(folder_selected))

            color_npz_path = os.path.join(os.path.join(folder_selected, file_split[0]), color_npz)
            depth_npz_path = os.path.join(os.path.join(folder_selected,file_split[0]), depth_npz)

            save_path = os.path.join(r'C:\Users\ADMIN\Desktop\check_folder',save_as)
            depth_image = np.load(depth_npz_path)['data']
            color_image = np.load(color_npz_path)['data']
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            print(depth_colormap.shape)

            images = np.hstack((color_image, depth_colormap))
            cv2.imwrite(save_path, depth_colormap)
            count+=1

    return count


def main():
    cwd = os.getcwd()
    count = convert_depth_images(cwd)
    print("Processed image count", count)

if __name__ == '__main__':
    main()
