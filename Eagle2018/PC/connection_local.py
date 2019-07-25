import paramiko
import os
import argparse

#服务器账户
ServerHostname = ''
ServerPort = '22'
ServerUsername = ''
ServerPassword = ''

parser = argparse.ArgumentParser()
parser.add_argument('--hostname', type = str, default = '10.101.169.60', help = 'Hostname')
parser.add_argument('--port', type = str, default = '22', help = 'Port')
parser.add_argument('--username', type = str, default = 'cuihaoran', help = 'Username')
parser.add_argument('--passcode', type = str, default = '981216', help = 'Passcode')
parser.add_argument('--dataset', type = str, default = 'Potsdam', help = 'Dataset')
parser.add_argument('--first', type = str, default = None, help = 'test 1')
parser.add_argument('--second', type = str, default = None, help = 'test 2')
parser.add_argument('--third', type = str, default = None, help = 'test 3')
parser.add_argument('--fourth', type = str, default = None, help = 'test 4')
#parser.add_argument()  add following attributes
arg = parser.parse_args()
record= []
record.append(arg.dataset)
record.append(arg.first)
record.append(arg.second)
record.append(arg.third)
record.append(arg.fourth)
#record.append following attributes
args = []

for i, val in enumerate(record):
	if val != None:
		args.append(val)
	else:
		args.append(-1) #if the arrtibutes are not required then set it to -1


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#ssh.connect(hostname='10.101.160.69', port=22, username='jiangyl', password='jiangyulai')
ssh.connect(hostname=ServerHostname, port=ServerPort, username=ServerUsername, password=ServerPassword)

#folder = '/server_space/jiangyl/dlink_potsdam_normto1/remote' #folder has script
#file = 'test_remote.py' #script name
folder = '/home/jiangyl/PDB/' #folder has script
file = 'connection_remote.py' #script name

#stdin, stdout, stderr = ssh.exec_command('source activate my_tensorflow; cd %s; python %s --dataset %s --first %0.1f --second %0.1f --third %0.1f --fourth %0.1f'%(folder, file, args[0], args[1], args[2], args[3], args[4]))
stdin, stdout, stderr = ssh.exec_command('cd %s;/home/jiangyl/.conda/envs/my_tensorflow/bin/python2.7 %s --hostname %s --port %s --username %s --passcode %s --dataset %s --first %s --second %s --third %s --fourth %s'%(folder, file, arg.hostname, arg.port, arg.username, arg.passcode, args[0], args[1], args[2], args[3], args[4]))

#print('cd %s;/home/jiangyl/.conda/envs/my_tensorflow/bin/python2.7 %s --dataset %s --first %s --second %s --third %s --fourth %s'%(folder, file, args[0], args[1], args[2], args[3], args[4]))


for s in stdout.readlines():
	print(s)
for s in stderr.readlines():
	print(s)
print("finish")
ssh.close()




