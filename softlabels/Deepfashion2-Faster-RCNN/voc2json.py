import os
import re
import json
import xmltodict

rootDir = "./pascal_annos"

def generateVOC2Json(rootDir,xmlFiles):
	attrDict = dict()
	#images = dict()
	#images1 = list()
	attrDict["categories"]=[{"supercategory":"clothes","id":1,"name":"short_sleeved_shirt"},
			        {"supercategory":"clothes","id":2,"name":"long_sleeved_shirt"},
			        {"supercategory":"clothes","id":3,"name":"short_sleeved_outwear"},
			        {"supercategory":"clothes","id":4,"name":"long_sleeved_outwear"},
				{"supercategory":"clothes","id":5,"name":"vest"},
				{"supercategory":"clothes","id":6,"name":"sling"},
				{"supercategory":"clothes","id":7,"name":"shorts"},
				{"supercategory":"clothes","id":8,"name":"trousers"},
				{"supercategory":"clothes","id":9,"name":"skirt"},
                {"supercategory":"clothes","id":10,"name":"short_sleeved_dress"},
                {"supercategory":"clothes","id":11,"name":"long_sleeved_dress"},
				{"supercategory":"clothes","id":12,"name":"vest_dress"},
				{"supercategory":"clothes","id":13,"name":"sling_dress"},
			      ]
	images = list()
	annotations = list()
	for root, dirs, files in os.walk(rootDir):
		image_id = 0
		for file in xmlFiles:
			image_id = image_id + 1
			annotation_path = os.path.abspath(file)
				
			#tree = ET.parse(annotation_path)#.getroot()
			image = dict()
			#keyList = list()
			doc = xmltodict.parse(open(annotation_path).read())
			#print doc['annotation']['filename']
			image['file_name'] = str(doc['annotation']['filename'])
			#keyList.append("file_name")
			image['height'] = int(doc['annotation']['size']['height'])
			#keyList.append("height")
			image['width'] = int(doc['annotation']['size']['width'])
			#keyList.append("width")
			#image_id = str(doc['annotation']['filename']).split('.png')[0]
			image['id'] = image_id
			print ("File Name: {} and image_id {}".format(file, image_id))
			images.append(image)
			# keyList.append("id")
			# for k in keyList:
			# 	images1.append(images[k])
			# images2 = dict(zip(keyList, images1))
			# print images2
			#print images

			#attrDict["images"] = images

			#print attrDict
			#annotation = dict()
			id1 = 1
			if 'object' in doc['annotation']:
				for value in attrDict["categories"]:
					annotation = dict()
					if str(doc['annotation']['object']['name']) == value["name"]:
						#print str(obj['name'])
						#annotation["segmentation"] = []
						annotation["iscrowd"] = 0
						#annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
						annotation["image_id"] = image_id
						x1 = int(doc['annotation']['object']["bndbox"]["xmin"]) 
						y1 = int(doc['annotation']['object']["bndbox"]["ymin"])
						x2 = int(doc['annotation']['object']["bndbox"]["xmax"])
						y2 = int(doc['annotation']['object']["bndbox"]["ymax"])
						annotation["bbox"] = [x1, y1, x2, y2]
						annotation["area"] = x2 * y2
						annotation["category_id"] = value["id"]
						annotation["ignore"] = 0
						annotation["id"] = str(image_id) + "-" + str(id1)
						id1 +=1

						annotations.append(annotation)
				
			else:
				print ("File: {} doesn't have any object".format(file))
			#image_id = image_id + 1
			

	attrDict["images"] = images	
	attrDict["annotations"] = annotations
	attrDict["type"] = "instances"

	#print attrDict
	jsonString = json.dumps(attrDict)
	with open("receipts_valid.json", "w") as f:
		f.write(jsonString)

import os, fnmatch
def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)

# rootDir = "/netscratch/pramanik/OBJECT_DETECTION/detectron/lib/datasets/data/Receipts/Annotations"
# for root, dirs, files in os.walk(rootDir):
# 	for file in files:
# 		if file.endswith(".xml"):
# 			annotation_path = str(os.path.abspath(os.path.join(root,file)))
# 			#print(annotation_path)
# 			generateVOC2Json(annotation_path)
trainXMLFiles = [rootDir + '/' + f for f in os.listdir(rootDir) if re.match(r'.+\.xml', f)]
findReplace(rootDir, "short-sleeve-top", "short_sleeved_shirt", "*.xml")
generateVOC2Json(rootDir, trainXMLFiles)