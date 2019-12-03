import os
import shutil
import xml.etree.ElementTree as ET

from enum import Enum

from collections import defaultdict
import json
import re

class DF2(Enum):
    short_sleeve_top = 1
    long_sleeve_top = 2
    short_sleeve_outwear = 3
    long_sleeve_outwear = 4
    vest = 5
    sling = 6
    shorts = 7
    trousers = 8
    skirt = 9
    short_sleeve_dress = 10
    long_sleeve_dress = 11
    vest_dress = 12
    sling_dress = 13


df2_enum_to_name = {
    DF2.short_sleeve_top: "short_sleeve_top",
    DF2.long_sleeve_top: "long_sleeve_top",
    DF2.short_sleeve_outwear: "short_sleeve_outwear",
    DF2.long_sleeve_outwear: "long_sleeve_outwear",
    DF2.vest: "vest",
    DF2.sling: "sling",
    DF2.shorts: "shorts",
    DF2.trousers: "trousers",
    DF2.skirt: "skirt",
    DF2.short_sleeve_dress: "short_sleeve_dress",
    DF2.long_sleeve_dress: "long_sleeve_dress",
    DF2.vest_dress: "vest_dress",
    DF2.sling_dress: "sling_dress",
}

discard_item = "discard_this"


class_conversion_table = {
    " skirts": DF2.skirt,
    " pants": DF2.trousers,
    "short-sleeve-dress": DF2.short_sleeve_dress,
    "shorts": DF2.shorts,
    "long_shirt": DF2.long_sleeve_top,
    "long_women_dress": DF2.long_sleeve_dress,
    "short-sleeve-top": DF2.short_sleeve_top,
    "long_women_top": DF2.long_sleeve_top,
    "long_top": DF2.long_sleeve_top,
    "vests": DF2.vest,
    "long_sleeved_shirt": DF2.long_sleeve_top,
    "Long sleeve outerwear": DF2.long_sleeve_outwear,
    "skirt": DF2.skirt,
    "vest_dress": DF2.vest_dress,
    "Long sleeve top": DF2.long_sleeve_top,
    "sling_dress": DF2.sling_dress,
    "short_women_dress": DF2.short_sleeve_dress,
    "sling": DF2.sling,
    "trousers": DF2.trousers,
    "long_sleeved_outwear": DF2.long_sleeve_outwear,
    " long_outwear": DF2.long_sleeve_outwear,
    "Short sleeve top-men": DF2.short_sleeve_top,
    "short_sleeved_shirt": DF2.short_sleeve_top,
    "short_sleeved_dress": DF2.short_sleeve_dress,
    "short_women_top": DF2.short_sleeve_top,
    "long-sleeve-top": DF2.long_sleeve_top,
    "Long sleeve shirt": DF2.long_sleeve_top,
    " long_women_top": DF2.long_sleeve_top,
    "onesies": discard_item,
    "pants": DF2.trousers,
    "long-sleeve-dress": DF2.long_sleeve_dress,
}


def get_class_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for child in root:
        if child.tag == "object":
            clazz = child[0]

            return clazz.text


def parse_xml(path, storage):
    storage["xml_paths"].append(path)

    clazz = get_class_from_xml(path)
    storage["classes"].add(clazz)


def parse_png(path, storage):
    storage["png_paths"].append(path)


def parse_folder(folder_path, storage=None):
    if storage is None:
        storage = {"xml_paths": list(), "png_paths": list(), "classes": set()}

    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)

        if os.path.isdir(path):
            parse_folder(path, storage=storage)

        elif ".xml" in path:
            parse_xml(path, storage)

        elif ".png" in path:
            parse_png(path, storage)

    return storage


def generate_dataset(xml_png_pairs):
    root_path = os.getcwd()
    data_path = os.path.join(root_path, "data")

    if not os.path.exists(data_path):
        os.mkdir("data")

    for c in DF2:
        class_name = c.name
        class_path = os.path.join(data_path, class_name)

        if not os.path.exists(class_path):
            os.mkdir(class_path)

    class_count = {c: 0 for c in DF2}
    unique_item_count = {}
    unique_item_paths = {}

    for xml_path, png_path in xml_png_pairs:
        raw_class = get_class_from_xml(xml_path)
        clean_class = class_conversion_table[raw_class]

        if clean_class == discard_item:
            continue

        class_count[clean_class] += 1

        xml_fn = os.path.split(xml_path)[1]
        png_fn = os.path.split(png_path)[1]

        clothing_id = f'{xml_fn.split("_")[0]}_{xml_fn.split("_")[1]}'
        clothing_id = re.sub('[,-]','',clothing_id)

        if clothing_id in unique_item_count:
            unique_item_count[clothing_id] += 1
            unique_item_paths[clothing_id] += [(xml_path, png_path)]
        else:
            unique_item_count[clothing_id] = 1
            unique_item_paths[clothing_id] = [(xml_path, png_path)]

        class_path = df2_enum_to_name[clean_class]
        new_xml_path = os.path.join(data_path, class_path, xml_fn)
        new_png_path = os.path.join(data_path, class_path, png_fn)

        # shutil.copy(xml_path, new_xml_path)
        # shutil.copy(png_path, new_png_path)

    print(class_count)

    for key, value in class_count.items():
        print(key.name, value)

    print(unique_item_count)
    print(len(unique_item_count.items()))

    incorrect = 0
    values = {}
    for key, value in unique_item_count.items():

        if str(value) in values:
            values[str(value)] += 1
        else:
            values[str(value)] = 1

        if value != 10:
            incorrect += 1

            print(key, value)
            for a, b in unique_item_paths[key]:
                print(a, b)


    print("incorrect: ", incorrect)
    for k, v in values.items():
        print(k, v)


def find_pairs(xml_paths, png_paths):
    matched_xml = [False for _ in xml_paths]
    matched_png = [False for _ in png_paths]

    matched = list()

    for xidx, xml_full_path in enumerate(xml_paths):
        xml_fn = os.path.split(xml_full_path)[1]
        xml_name = os.path.splitext(xml_fn)[0]

        for pidx, png_full_path in enumerate(png_paths):
            png_fn = os.path.split(png_full_path)[1]
            png_name = os.path.splitext(png_fn)[0]

            if xml_name == png_name:
                if matched_png[pidx]:
                    print("FOUND DUPLICATE")
                    print(xml_full_path, png_full_path, sep="\n")
                    exit()

                matched.append((xml_full_path, png_full_path))

                matched_xml[xidx] = True
                matched_png[pidx] = True

                break

    non_matched = list()

    def add_non_matched(match_list, item_list):
        for idx, found in enumerate(match_list):
            if not found:
                non_matched.append(item_list[idx])

    add_non_matched(matched_xml, xml_paths)
    add_non_matched(matched_png, png_paths)

    return matched, non_matched


def labels_to_DF_format(xml_paths):

    invalid_labels = []
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        object_ = root.find('object')
        name_ = object_.find("name")
        clean_tag = ET.SubElement(object_,'clean_label')

        clean_tag.text = "1" 
        old_label = name_.text
        clean_label = class_conversion_table.get(old_label)
        new_label = df2_enum_to_name.get(clean_label)
        if clean_label != None and new_label != None:
            name_.text = new_label
        else:
            invalid_labels.append(old_label)

        # Overwrite XML file with the new labels
        tree.write(xml_path)

    print(f"Labels not valid : {set(invalid_labels)}")

def main():
    s = parse_folder("/home/datta/lab/_KTH_ACADEMIA/pj_ds/")

    xml_paths = s["xml_paths"]
    png_paths = s["png_paths"]
    classes = s["classes"]

    labels_to_DF_format(xml_paths)
    exit()

    print(len(xml_paths))
    print(len(png_paths))

    print(classes)
    print(len(classes))

    paired, non_paired = find_pairs(xml_paths, png_paths)

    print("paired", len(paired))
    print("non_paired", len(non_paired))

    for lonely_file in non_paired:
        print(lonely_file)

    if len(non_paired) == 0:
        generate_dataset(paired)


if __name__ == "__main__":
    main()
