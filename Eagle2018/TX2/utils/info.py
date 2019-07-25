import os

dir = "/home/jiangyl/pics/text.txt"
#dir = "D:\\study\\Eagle\\DB\\pics\\test2.txt"
file = open(dir, 'r')

for line in file:
    info = line.split(' ')
    newFile = open(info[6].st, 'w')
    newFile.writelines('房屋占地比： ' + info[0])
    newFile.writelines('草地占地比： ' + info[1])
    newFile.writelines('汽车数量： ' + info[2])

newFile.close()

file.close()