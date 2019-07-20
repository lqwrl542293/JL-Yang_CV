#!/usr/bin/env python
# coding=utf-8
import glob
from PIL import Image
import os

classes = ["aeroplane", "ship", "storage_tank", "baseball_diamond", "tennis_court", "basketball_court", "ground_track_field", "harbor", "bridge", "vehicle"]

s1="""	<object>
		<name>{0}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{1}</xmin>
			<ymin>{2}</ymin>
			<xmax>{3}</xmax>
			<ymax>{4}</ymax>
		</bndbox>
	</object>"""

s2="""<annotation>
	<folder>VOC2007</folder>
	<filename>{0}</filename>
	<path>{1}</path>
	<source>
		<database>My Database</database>
		<annotation>VOC2007</annotation>
		<image>flickr</image>
		<flickrid>NULL</flickrid>
	</source>
	<owner>
		<flickrid>NULL</flickrid>
		<name>J</name>
	</owner>
	<size>
		<width>{2}</width>
		<height>{3}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	{4}
</annotation>
"""

textlist=glob.glob('./ground_truth/*.txt')
for text_ in textlist:
    flabel = open(text_, 'r')
    lb = flabel.readlines()
    flabel.close()
    str = ""
    filename = text_[-7:-4]
    for i in range(0, len(lb)):
        tmp = lb[i].split(',')
        xmin = tmp[0][1:]
        ymin = tmp[1][:-1]
        xmax = tmp[2][1:]
        ymax = tmp[3][:-1]
        name = classes[int(tmp[4]) - 1]
        str += '\n' + s1.format(name, xmin, ymin, xmax, ymax)

    img = Image.open("./JPEGImages/" + filename + ".jpg")
    imageSize = img.size
    width = imageSize[0]
    height = imageSize[1]
    imagename = filename + ".jpg"
    path = os.getcwd()
    output = s2.format(imagename, path + '/' +imagename , width, height, str)
    savename='Annotations/' + filename + '.xml'
    f = open(savename, 'w')
    f.write(output)
    f.close()
