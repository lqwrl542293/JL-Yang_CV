import os

rootdir = '/home/jiangyl/pics'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件

for i in range(0, len(list)):
    path = os.path.join(rootdir, list[i])
    #print(path)
    if path.find('RGB') != (-1):
        path1 = path.replace('jpg', 'png')
        print(path)
        print(path1)
        os.rename(path, path1)

        
'''for i in range(0, len(list)):
    path = os.path.join(rootdir, list[i])
    #print(path)
    if path.find('txt?') != (-1):
        path1 = path.replace('txt?', 'txt')
        print(path)
        print(path1)
        os.rename(path, path1)'''