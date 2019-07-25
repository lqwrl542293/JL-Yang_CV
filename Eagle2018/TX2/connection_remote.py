import os
import argparse
import paramiko

parser = argparse.ArgumentParser()
parser.add_argument('--hostname', type = str, default = '', help = 'Hostname')
parser.add_argument('--port', type = str, default = '22', help = 'Port')
parser.add_argument('--username', type = str, default = '', help = 'Username')
parser.add_argument('--passcode', type = str, default = '', help = 'Passcode')
parser.add_argument('--dataset', type = str, default = 'Potsdam', help = 'Dataset')
parser.add_argument('--first', type = str, default = '-1', help = 'test 1')
parser.add_argument('--second', type = str, default = '-1', help = 'test 2')
parser.add_argument('--third', type = str, default = '-1', help = 'test 3')
parser.add_argument('--fourth', type = str, default = '-1', help = 'test 4')
#parser.add_argument()  add following attributes

arg = parser.parse_args()
#dirc = '/server_space/jiangyl/dlink_potsdam_normto1/output/' #database directory
dirc = '/home/jiangyl/PDB/'
folder = '/mnt/d/study/eagle/DB/DB/' #CHR

record = []
record.append(arg.dataset)
record.append(arg.first)
record.append(arg.second)
record.append(arg.third)
record.append(arg.fourth)
#args.append following attributes
#print(arg.first)

args = []
target = []

for i in range(0, len(record) - 1):

        if record[i+1] != '-1':
                args.append(record[i+1])
        else:
                args.append('-1')


if record[0] == 'Potsdam':
        directory = os.path.join(dirc, 'PDB.txt') #1.txt can be replaced by the name of database
#if record[1] == 'xxx' add other databases
#print('directory: '+directory)

lines = open(directory).readlines()
judge = 0

for line in lines:
        #splits = line.replace('\n', '').split('\t')
        splits = line.split(' ')
        #print(splits)
        for i, val in enumerate(args):
                #print('val:')
                #print(i,val)
                if val != '-1':
                        #print(val)
                        num = str(val).split(',')
                        #minnum = num[0]
                        #print('num')
                        #print(num)
                        maxnum=num[1]
                        minnum=num[0]
                        #print(splits[i])
                        if float(splits[i]) >= float(minnum) and float(splits[i]) <= float(maxnum):
                                judge = 1
                                #print('suitable')
                        else:
                                judge = 0
                        if judge == 0:
                                break;
        if judge == 1:
                #print('split')
                #print(splits)
                target.append(splits[3])
                target.append(splits[4])
                target.append(splits[5])
                target.append(splits[6].replace('\n',''))

#print(target)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(hostname=arg.hostname, port=arg.port, username=arg.username, password=arg.passcode)
#folder = '~/Downloads/receive/' #the directory used to save target_pic
#folder = '/Users/zhengyumin/Downloads/receive/'
#folder = '/home/zhuqingtian/db/' #ZQT
dirc = '/home/jiangyl/PDB/pics/'
#folder = '/mnt/d/study/eagle/DB/DB/' #CHR
#print(target)
sftp = ssh.open_sftp()
for i, pic in enumerate(target):
        #print('remotefile\n'+folder+str(i)+'.bmp')
        #sftp.put(pic, folder+str(i)+'.png')
        sftp.put(pic,pic.replace(dirc, folder))
        print(pic.replace(dirc, folder).replace('/mnt/d', 'D:/'))
        #print(folder+str(i)+'.png')
        #sftp.put(pic, folder)
       # sftp.put('/server_space/jiangyl/dlink_potsdam_normto1/output/3/top_potsdam_2_13_rgb-14.bmp', '~/Downloads/1.bmp')
sftp.close()
        #sftp.put(pic, folder)
