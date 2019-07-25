import os

#dir = "txt文件路径"
fileName = "info.txt"
dir = "C:\\Users\\34322\\Desktop\\results\\results"

filenames = os.listdir(dir) #获取文件夹下所有文件

filenameAb = dir + '\\' + fileName #绝对路径
file = open(filenameAb, 'w') #新建合并后的txt

for filename in filenames:
    filenameAb= dir + '\\' + filename #绝对路径
    fileData = ""
    with open(filenameAb, 'r') as f:
        for line in f:
            line = filename.strip('.txt') + ' ' + line
            file.writelines(line)
    file.write('\n')
        #fileData += line
   #file.write(fileData)
file.close()