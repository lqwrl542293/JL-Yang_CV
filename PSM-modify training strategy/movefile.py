import shutil
import argparse
import os
from dataloader import KITTIloader2015 as ls

parser = argparse.ArgumentParser(description='movefile')
parser.add_argument('--datapath',default='./training/')
parser.add_argument('--toWhere',default='./gt/')
args = parser.parse_args()

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

for index in range(len(test_left_disp)):
    shutil.move(test_left_disp[index], args.toWhere)
    print(test_left_disp[index])
