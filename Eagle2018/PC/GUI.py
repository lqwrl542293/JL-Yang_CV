from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import Toplevel
from PIL import Image, ImageTk
import paramiko
#from tipWindows import TipWindows
import os
import socket

#服务器账户
ServerHostname = ''
ServerPort = '22' #default port
ServerUsername = ''
ServerPassword = ''

#个人用户
#获取本机电脑名
myname = socket.getfqdn(socket.gethostname())
#获取本机ip
myaddr = socket.gethostbyname(myname)
#由于实际连接ubuntu
myname = ''
myport = '22'
mypasscode = ''


class Application(Frame):

    className = []
    prob = []
    #image_path_temporary = "D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images"

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.window_init()
        self.grid()

    def window_init(self):
        self.master.title('Eagle')
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()  # 设置界面宽度为530，高度为365像素，并且基于屏幕居中
        width = 500
        height = 400
        size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.master.geometry(size)
        self.grid()
        self.create_widgets_initial()

    def create_widgets_initial(self):
        #self.frame = Frame(self)
        #self.frame.grid()
        self.tabControl = ttk.Notebook(self)
        self.tab1 = Frame(self.tabControl)
        self.tabControl.add(self.tab1, text='企业用户')
        self.tab2 = Frame(self.tabControl)
        self.tabControl.add(self.tab2, text='个人用户')
        self.tabControl.pack(expand=1, fill="both")
        self.create_widgets_user()
        self.create_widgets()

    def create_widgets(self):
        #self.frame.destroy()
        #self.frame1 = Frame(self)
        #self.frame1.grid()
        #self.frame2 = Frame(self)
        #self.frame2.grid()
        self.chooseLabel = Label(self.tab1, text='请选择进行所需图片的语义信息')
        self.chooseLabel.grid(column=1, row=1, sticky='W')
        #comvalue.append(print("comvalue" + str(i)))
        #comboxlist.append(print("comboxlist" + str(i)))
        #self.comvalue = StringVar()  # 窗体自带的文本，新建一个值
        self.comboxlist=ttk.Combobox(self.tab1, textvariable='', state="readonly") #初始化
        self.comboxlist["values"] = ("房屋占地率(%)", "绿化率(%)", "车辆数目", "篮球场数目")
        #self.comboxlist["values"] = ("first", "second", "third", "fourth")
        self.comboxlist.current(0)
        self.comboxlist.bind("<<ComboboxSelected>>", print(self.comboxlist.get()))  #绑定事件,(下拉列表框被选中时，绑定go()函数)
        self.comboxlist.grid(row=2, column=0)
        self.spinBox1 = Spinbox(self.tab1, from_=0, to=100, increment=1)
        self.spinBox1.grid(row=2, column=1)
        self.spinBox2 = Spinbox(self.tab1, from_=0, to=100, increment=1)
        self.spinBox2.grid(row=2, column=2)
        self.addButton = Button(self.tab1, text='确定并增加条件', command=self.add_condition)
        self.addButton.grid(row=3, column=1)
        self.cancelButton = Button(self.tab1, text='条件输入有误，重新输入', command=self.delete_condition)
        self.cancelButton.grid(row=4, column=1)
        self.confirmButton = Button(self.tab1, text='完成所有条件', command=self.check)
        self.confirmButton.grid(row=5, column=1)
        #self.backButton = Button(self.tab1, text='返回初始界面', command=self.create_widgets_initial)
        #self.backButton.grid()
        self.quitButton = Button(self.tab1, text='Quit', command=self._quit)
        self.quitButton.grid(row=6, column=1)

    def add_condition(self):
        self.confirm_condition()
        self.addButton.destroy()
        self.cancelButton.destroy()
        self.confirmButton.destroy()
        self.quitButton.destroy()
        #self.comvalue = StringVar()  # 窗体自带的文本，新建一个值
        self.comboxlist = ttk.Combobox(self.tab1, textvariable='')  # 初始化
        self.comboxlist["values"] = ("房屋占地率(%)", "绿化率(%)", "车辆数目", "篮球场数目")
        #self.comboxlist["values"] = ("first", "second", "third", "fourth")
        self.comboxlist.current(0)
        self.comboxlist.bind("<<ComboboxSelected>>", print(self.comboxlist.get()))  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
        self.comboxlist.grid(row=len(self.className)+2, column=0)
        self.spinBox1 = Spinbox(self.tab1, from_=0, to=100, increment=1)
        self.spinBox1.grid(row=len(self.className)+2, column=1)
        self.spinBox2 = Spinbox(self.tab1, from_=0, to=100, increment=1)
        self.spinBox2.grid(row=len(self.className) + 2, column=2)
        self.addButton = Button(self.tab1, text='确定并增加条件', command=self.add_condition)
        self.addButton.grid(row=len(self.className)+3, column=1)
        self.cancelButton = Button(self.tab1, text='条件输入有误，重新输入', command=self.delete_condition)
        self.cancelButton.grid(row=len(self.className)+4, column=1)
        self.confirmButton = Button(self.tab1, text='完成所有条件', command=self.check)
        self.confirmButton.grid(row=len(self.className)+5, column=1)
        # self.backButton = Button(self.tab1, text='返回初始界面', command=self.create_widgets_initial)
        # self.backButton.grid()
        self.quitButton = Button(self.tab1, text='Quit', command=self._quit)
        self.quitButton.grid(row=len(self.className)+6, column=1)
        #self.addButton.grid(row=len(self.className)+1, column=2)

    def confirm_condition(self):
        #print("confirm")
        if(self.comboxlist.get() in self.className):
            self._msgbox("条件已存在，请重新选择")
            return 0
        else:
            self.className.append(self.comboxlist.get())
            self.prob.append(self.spinBox1.get())
            self.prob.append(self.spinBox2.get())
            return 1

    def delete_condition(self):
        if(messagebox.askyesno(message='确定清除所有条件吗')):
            for i in range(0, len(self.className)):
                self.className.clear()
                self.prob.clear()
            # self.comboxlist.destroy()
            # self.spinBox.destroy()
            # self.addButton.destroy()
            #self.lab1.destroy()
            self.tab1.destroy()
            self.tab2.destroy()
            #self.tab1 = Frame(self.tabControl)
            #self.tabControl.add(self.tab1, text='企业用户')
            self.create_widgets_initial()

    def check(self):
        if (self.confirm_condition() == 0):
            self.comboxlist = ttk.Combobox(self.tab1, textvariable='')  # 初始化
            self.comboxlist["values"] = ("房屋占地率(%)", "绿化率(%)", "车辆数目", "篮球场数目")
            #self.comboxlist["values"] = ("first", "second", "third", "fourth")
            self.comboxlist.current(0)
            self.comboxlist.bind("<<ComboboxSelected>>", print(self.comboxlist.get()))  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
            self.comboxlist.grid(row=len(self.className) + 1, column=0)
            self.spinBox1 = Spinbox(self.tab1, from_=0, to=100, increment=1)
            self.spinBox1.grid(row=len(self.className) + 1, column=1)
            self.spinBox2 = Spinbox(self.tab1, from_=0, to=100, increment=1)
            self.spinBox2.grid(row=len(self.className) + 1, column=2)
            self.addButton = Button(self.tab1, text='确定并增加条件', command=self.add_condition)
            self.addButton.grid(row=len(self.className) + 2, column=1, columnspan=2)
        else:
            self.addButton.destroy()
            self.confirmButton.destroy()
            self.cancelButton.destroy()
            self.quitButton.destroy()
            self.cancelButton = Button(self.tab1, text='条件输入有误，重新输入', command=self.delete_condition)
            self.cancelButton.grid(row=len(self.className) + 2, column=1)
            self.runButton = Button(self.tab1, text='开始运行', command=self.run)
            self.runButton.grid(row=len(self.className) + 3, column=1)
            self.quitButton = Button(self.tab1, text='Quit', command=self._quit)
            self.quitButton.grid(row=len(self.className) + 4, column=1)

    def run(self):
        self.addButton.destroy()
        self.cancelButton["text"] = "清除所有条件"
        self.info = []

        #合成命令

        #激活环境并运行
        self.info.append("cd venv\Scripts\&activate.bat&cd ..&cd ..&python connection_local.py")
        #个人用户电脑信息
        self.info.append(" --hostname " + myaddr + " --port " + myport + " --username " + myname + " --passcode " + mypasscode)
        #需要提取的信息条件
        for i in range(0, len(self.className)):
            self.info.append(" --" + self.className[i] + " " + self.prob[2*i] + "," + self.prob[2*i+1])
        self.info_str = ''.join(self.info)
        #print(self.info)
        print(self.info_str)
        self.info_str = self.info_str.replace("房屋占地率(%)", "first")
        self.info_str = self.info_str.replace("绿化率(%)", "second")
        self.info_str = self.info_str.replace("车辆数目", "third")
        self.info_str = self.info_str.replace("篮球场数目", "fourth")
        print(self.info_str)

        path = 'D:\study\Eagle\DB\DB'
        for i in os.listdir(path):
            path_file1 = os.path.join(path, i)
            if os.path.isfile(path_file1):
                os.remove(path_file1)
        #清空路径下的文件


        output = os.popen(self.info_str)
        op = output.read()
        print(op)
        img_path = []
        for line in op.splitlines():
            img_path.append(line)
            if img_path[len(img_path) - 1] == "finish":
                print(img_path)
                img_path1 = os.path.abspath(os.path.dirname(img_path[0]))
                img_path1 = img_path1.replace('\mnt\d', '')
                self.result(img_path1)
                self.pic_win.resultLabel["text"] = "满足条件图片已保存在本地文件夹：\n" + img_path1
                self.pic_win.canvas = Canvas(self.pic_win)
                self.pic_win.canvas.pack()
                self.pic_win.checkButton = Button(self.pic_win, text="查看图片", command=lambda: self.pic_preview(img_path1))
                self.pic_win.checkButton.pack()
        img_path=['D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-021.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-035.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-038.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-040.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-021.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-035.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-038.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-040.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-021.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-035.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-038.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-040.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-021.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-035.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-038.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-040.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-021.png','D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-035.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-038.png', 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images\SNAG-040.png']
        img_path1 = 'D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images'
        self.result(img_path1)

    def result(self, img_path1):
        self.pic_win = Toplevel()
        screenwidth = self.pic_win.winfo_screenwidth()
        screenheight = self.pic_win.winfo_screenheight()  # 设置界面宽度为530，高度为365像素，并且基于屏幕居中
        width = 1500
        height = 800
        size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.pic_win.geometry(size)
        self.pic_win.grid()
        self.pic_win.title("结果")
        self.pic_win.resultLabel = Label(self.pic_win, text="满足条件图片已保存在本地文件夹：\n"+img_path1)
        self.pic_win.resultLabel.pack()
        self.pic_win.checkButton = Button(self.pic_win, text="查看图片", command=lambda: self.pic_preview(img_path1))
        self.pic_win.checkButton.pack()
        self.listBox = Listbox(self.pic_win)
        self.scrollbar = Scrollbar(self.listBox, orient=HORIZONTAL)
        self.scrollbar.pack(side=BOTTOM, fill="x")
        self.listBox.config(xscrollcommand=self.scrollbar.set)
        self.scrollbar.set(0.6, 0)
        self.scrollbar.config(command=self.listBox.xview())
        gifsdict = {}
        self.listBox.pack(expand = YES)
        self.pic_win.infoLabel = Label(self.pic_win, text='info:', justify=LEFT)
        self.pic_win.infoLabel.pack()

    def pic_preview(self, img_path1):#父窗口可以在主窗口之前关的问题为解决
        self.pic_win.wm_attributes('-topmost', 1)
        self.pic_win.wm_attributes('-topmost', 0)
        #print("sssss"+img_path1)
        self.filePath = filedialog.askopenfilename(parent=self, initialdir=img_path1, title='Choose an image.')
        odFilePath = str(self.filePath).replace('RGB', 'OD')
        ssFilePath = str(self.filePath).replace('RGB', 'SS')
        infoFilePath = str(self.filePath).replace('RGB.png', 'INFO.txt')
        img = Image.open(self.filePath)
        filename = ImageTk.PhotoImage(img)
        odImg = Image.open(odFilePath)
        odFilename = ImageTk.PhotoImage(odImg)
        ssImg = Image.open(ssFilePath)
        ssFilename = ImageTk.PhotoImage(ssImg)
        self.pic_win.wm_attributes('-topmost', 1)
        #Label_image = Image.open(self.filePath)
        #Label_image.show()
        #self.pic_win.canvas.image = filename  # <--- keep reference of your image
        #self.pic_win.canvas.create_image(0, 0, anchor='nw', image=filename)
        self.imageLabel = Label(self.listBox, imag=filename)
        self.imageLabel.pack(side="left", fill="both", expand="yes")
        self.odLabel = Label(self.listBox, imag=odFilename)
        self.odLabel.pack(side="left", fill="both", expand="yes")
        self.ssLabel = Label(self.listBox, imag=ssFilename)
        self.ssLabel.pack(side="left", fill="both", expand="yes")
        infoFile = open(infoFilePath, encoding='UTF-8')
        infoList = []
        for line in infoFile.readlines():
            infoList.append(line)
        info = ''.join(infoList)
        print(info)
        self.pic_win.infoLabel = Label(self.pic_win, text=info, justify=LEFT)
        self.pic_win.infoLabel.pack()
        self.listBox.insert(ACTIVE, self.imageLabel)
        self.listBox.insert(ACTIVE, self.odLabel)
        self.listBox.insert(self.ssLabel)
        #self.listBox.insert(END, self.infoLabel)
        self.listBox.pack()
        self.pic_win.checkButton["text"] = "查看下一张图片"


    def _msgbox(self, msg):
        messagebox.showinfo('提示', msg)


    def create_widgets_user(self):
        #self.frame.destroy()
        #self.frame3 = Frame(self)
        #self.frame3.grid()
        self.chooseLabel = Label(self.tab2, text='请选择进行目标检测和语义分割的图片')
        self.chooseLabel.pack()
        self.canvas = Canvas(self.tab2, height=200, width=200)
        self.canvas.pack()
        self.pathLabel = Label(self.tab2, text='Image Path')
        self.pathLabel.pack()
        self.chooseButton = Button(self.tab2, text='Choose', command=self.pic_choose_user)
        self.chooseButton.pack()
        self.detectButton = Button(self.tab2, text='Detect', command=self.detect_user)
        self.detectButton.pack()
        self.quitButton = Button(self.tab2, text='Quit', command=self._quit)
        self.quitButton.pack()

    def pic_choose_user(self):
        self.filePath = filedialog.askopenfilename(parent=self, initialdir="C:/", title='Choose an image.')
        #self.filePath = filedialog.askopenfilename(parent=self, initialdir=self.image_path_temporary, title='Choose an image.')
        img = Image.open(self.filePath)
        img_width = img.size[0]
        img_height = img.size[1]
        rate = img_width/img_height
        if rate > 1:
            rate1 = img_width / 200
            img_width = 200
            img_height = img_height / rate1
        else:
            rate1 = float(img_height) / 200.0
            img_height = 200
            img_width = img_width/rate1
        img_new = img.resize((int(img_width), int(img_height)))
        filename = ImageTk.PhotoImage(img_new)
        self.canvas.image = filename  # <--- keep reference of your image
        self.canvas.create_image(0, 0, anchor='nw', image=filename)

    def detect_user(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=ServerHostname, port=ServerPort, username=ServerUsername, password=ServerPassword)

        folder = '/server_space/jiangyl/dlink_potsdam_normto1/remote'  # folder has script
        file = 'predict_remote.py'

        sftp = ssh.open_sftp()
        sftp.put(self.filePath, folder + 'test.png')

        stdin, stdout, stderr = ssh.exec_command('cd %s; python %s ' % (folder, file))
        for s in stdout:
            if s == "finish":
                self.pic_preview_user()

        ssh.close()

        self.pic_preview_user("D:\study\Eagle\squeeze_yolo_gui\squeeze_yolo_gui\datasets\google_maps\images/001.png")

    def pic_preview_user(self, filePath):
        self.pic_win_user = Toplevel()
        self.pic_win_user.wm_attributes('-topmost', 1)
        self.pic_win_user.wm_attributes('-topmost', 0)
        self.listBox_user = Listbox(self.pic_win_user)
        self.scrollbar_user = Scrollbar(self.listBox_user, orient=HORIZONTAL)
        self.scrollbar_user.pack(side=BOTTOM, fill="x")
        self.listBox_user.config(xscrollcommand=self.scrollbar_user.set)
        self.scrollbar_user.set(0.6, 0)
        self.scrollbar_user.config(command=self.listBox_user.xview())
        gifsdict = {}
        self.listBox_user.pack(expand=YES)
        odFilePath = filePath.replace('RGB', 'OD')
        ssFilePath = filePath.replace('RGB', 'SS')
        infoFilePath = filePath.replace('RGB.png', 'INFO.txt')
        print(filePath)
        img = Image.open(filePath)
        filename = ImageTk.PhotoImage(img)
        odImg = Image.open(odFilePath)
        odFilename = ImageTk.PhotoImage(odImg)
        ssImg = Image.open(ssFilePath)
        ssFilename = ImageTk.PhotoImage(ssImg)
        self.pic_win_user.wm_attributes('-topmost', 1)
        self.imageLabel_user = Label(self.listBox_user, imag=filename)
        self.imageLabel_user.pack(side="left", fill="both", expand="yes")
        self.odLabel_user = Label(self.listBox_user, imag=odFilename)
        self.odLabel_user.pack(side="left", fill="both", expand="yes")
        self.ssLabel_user = Label(self.listBox_user, imag=ssFilename)
        self.ssLabel_user.pack(side="left", fill="both", expand="yes")
        infoFile = open(infoFilePath)
        infoList = []
        for line in infoFile.readlines():
            infoList.append(line)
        info = ''.join(infoList)
        print(info)
        self.infoLabel_user = Label(self.pic_win_user, text=info, justify=LEFT)
        self.infoLabel_user.pack()
        self.listBox_user.insert(ACTIVE, self.imageLabel_user)
        self.listBox_user.insert(ACTIVE, self.odLabel_user)
        self.listBox_user.insert(self.ssLabel_user)
        # self.listBox_user.insert(END, self.infoLabel)
        self.listBox_user.pack()


    def _quit(self):
        self.quit()
        self.destroy()
        exit()

if __name__ == '__main__':
    app = Application()
    # to do
    app.mainloop()