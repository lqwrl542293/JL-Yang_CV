import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import pickle
from tqdm import tqdm

# DATA_ROOT = r'F:\MachineLearning-Datasets\SelfBuildUAV\USSD'
DATA_ROOT = r'/root/Datasets/USSD'


class USSDataset(data.Dataset):
    def __init__(self, root, list_path='', max_iters=None, crop_size=(544, 960), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(DATA_ROOT, name.split(',')[0])
            label_file = osp.join(DATA_ROOT, name.split(',')[1])
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        # 随机resize到 [0.75, 2] 之间的某个scale
        if self.scale:
            scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # random resize between 0.5 and 2
            f_scale = scale[random.randint(0, 5)]
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image[:, :, ::-1]  # change to RGB
        img_h, img_w = label.shape

        # 进行pad(crop)
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        # 随机镜像
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1      # -1 / 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class USSDValDataset(data.Dataset):
    def __init__(self, root='', list_path='', f_scale=1, mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.f_scale = f_scale
        self.mean = mean
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(DATA_ROOT, name.split(',')[0])
            label_file = osp.join(DATA_ROOT, name.split(',')[1])
            # image_name = name.strip().split()[0].strip().split('/', 3)[3].split('.')[0]
            image_name = name.split(',')[0].split('/')[1].split('.')[0]
            print(image_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape
        name = datafiles["name"]
        if self.f_scale != 1:
            image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        # print('image.shape:',image.shape)
        return image.copy(), label.copy(), np.array(size), name


class USSDTrainInform:
    def __init__(self, data_dir='', classes=6,
                 train_set_file="", inform_data_file="", normVal=1.10):

        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, train_flag=True):
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(osp.join(self.data_dir, self.train_set_file)) as textFile:
            for line in tqdm(textFile):
                line_arr = line.split(',')
                img_file = (osp.join(DATA_ROOT, line_arr[0]))
                label_file = osp.join(DATA_ROOT, line_arr[1].strip())

                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag:
                    hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)

                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        # return_val = self.readWholeTrainSet(fileName=self.train_set_file)
        return_val = self.readWholeTrainSet()

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None


if __name__ == '__main__':
    # ussd = USSDataset(root='', list_path=r"F:\ZYHWorkSpace\BiSeNet_v1_v2_PyTorch\datasets\USSD\test_resized_2.txt")
    # ussd = USSDValDataset(root='',
    #                       list_path=r"F:\ZYHWorkSpace\BiSeNet_v1_v2_PyTorch\datasets\USSD\test_resized_2.txt")
    pass
