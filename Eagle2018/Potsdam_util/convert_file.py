# @Author: Yumin Zheng
# @File : multi_divide_pics.py
# @Version : 1.0
# @Last modified by : Yumin Zheng




#Potsdam dataset only. You can follow this src to match your own dataset



import multiprocessing
import os
import numpy as np
import cv2



def convert_file(filename):
    if filename[-3:] == "bmp":
        filepath = "xxx" + filename    #source folder path
        print(filename)
        mask = cv2.imread(filepath, 1)
        (b, g, r) = cv2.split(mask)
        mask = cv2.merge([r, g, b])
        label_map = np.zeros([mask.shape[0], mask.shape[1], 6])#(gt_width, gt_length, num_classes)
    # surfaces(RGB: 255, 255, 255)
    # Building(RGB: 0, 0, 255)
    # Low vegetation(RGB: 0, 255, 255)
    # Tree(RGB: 0, 255, 0)
    # Car(RGB: 255, 255, 0)
    # Clutter / background(RGB: 255, 0, 0)
        pic = mask
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                if pic[i][j][0] == 255 and pic[i][j][1] == 255 and pic[i][j][2] == 255:
                    label_map[i][j] = np.array([1, 0, 0, 0, 0, 0])
                elif pic[i][j][0] == 0 and pic[i][j][1] == 0 and pic[i][j][2] == 255:
                    label_map[i][j] = np.array([0, 1, 0, 0, 0, 0])
                elif pic[i][j][0] == 0 and pic[i][j][1] == 255 and pic[i][j][2] == 255:
                    label_map[i][j] = np.array([0, 0, 1, 0, 0, 0])
                elif pic[i][j][0] == 0 and pic[i][j][1] == 255 and pic[i][j][2] == 0:
                    label_map[i][j] = np.array([0, 0, 0, 1, 0, 0])
                elif pic[i][j][0] == 255 and pic[i][j][1] == 255 and pic[i][j][2] == 0:
                    label_map[i][j] = np.array([0, 0, 0, 0, 1, 0])
                elif pic[i][j][0] == 255 and pic[i][j][1] == 0 and pic[i][j][2] == 0:
                    label_map[i][j] = np.array([0, 0, 0, 0, 0, 1])
        label_map = label_map.transpose(2, 0, 1)
#   np.save(filepath.split('.')[0], label_map)


        np.save("xxx",label_map)# save path

if __name__ == '__main__':
    multiprocessing.freeze_support()
    files = os.listdir("xxx") #source folder
    p = multiprocessing.Pool(10) #enable multi-process
    p.map(convert_file, files)
