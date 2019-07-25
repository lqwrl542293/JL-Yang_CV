fileName = "C:\\Users\\34322\\Desktop\\results\\results\\comp4_det_test_vehicle.txt"
DB = "C:\\Users\\34322\\Desktop\\test.txt" #DB.txt的绝对路径
#需要先删除DB里最后一个换行符号

DBfile = open(DB, 'r')

file = open(fileName, 'r')

#classes = ["1", "2", "3", "4"]

#classed = ["vehicle"]

lines = ""

for DBline in DBfile:
    DBline = DBline.strip('\n')
    print(DBline)
    DBinfo = DBline.split('\t') #DB每个数之间的间隔
    print(DBinfo)
    print(DBinfo[2])
    picFile = DBinfo[2]
    print(picFile)
    #picName = picFile.strip('.png')
    #print(picName)

    #初始化类别计算的数
    class1 = 0
    class2 = 0
    class3 = 0
    class4 = 0

    for line in open(fileName, 'r'):
        info = line.split(' ')
        '''if info[1] == picFile:
            if info[0] == classes[0]:
                class1 += 1
            elif info[0] == classes[1]:
                class2 += 1
            elif info[0] == classes[2]:
                class3 += 1
            elif info[0] == classes[3]:
                class4 += 1'''
        if picFile.find(info[0]) != (-1):
            if float(info[1]) > 0.5:
                class1 += 1

    pic1 = DBinfo.pop().replace('pics', 'home/nvidia/pics')
    pic2 = DBinfo.pop().replace('pics', 'home/nvidia/pics')
    pic3 = DBinfo.pop().replace('pics', 'home/nvidia/pics')
    DBinfo[0] = DBinfo[0] + ' '
    DBinfo[1] = DBinfo[1] + ' '
    DBinfo.append(str(class1)+' ')
    #DBinfo.append(str(class2)+'\t')
    #DBinfo.append(str(class3)+'\t')
    #DBinfo.append(str(class4)+'\t')
    DBinfo.append(pic3+' ')
    DBinfo.append(pic2+' ')
    DBinfo.append(pic1+'\n')

    finalLine = ''.join(DBinfo)
    lines += finalLine

    #file.close()
DBfile.close()

with open(DB, 'w') as f:
    f.write(lines)

f.close()

'''for DBline in DBfile:
    DBline = DBline.strip('\n')
    print(DBline)'''