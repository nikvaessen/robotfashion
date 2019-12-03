import xml.etree.ElementTree as ET
import os

distinct_names = list()


def parse_files(folder_path):
    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)
        if ".xml" in path:
            tree = ET.parse(path)
            root = tree.getroot()
            for element in root:
                for enum, subelement in enumerate(element):

                    if enum == 0:
                        distinct_names.append(subelement.text)
                        # print(subelement.text)
                        # subelement.text = 'Long sleeve shirt'
            # tree.write(path)
            print(list(set(distinct_names)))


def main():
    cwd = os.getcwd()
    parse_files(cwd)


if __name__ == "__main__":
    main()
