"""
创建所需的image、label索引文件
"""
import os
import random


def create_index_file(data_root, img_dir, label_dir, save_path, subset='train'):
    img_path_prefix = os.path.join(subset, 'img-png')
    label_path_prefix = os.path.join(subset, 'label_gray')

    cnt = 0
    with open(save_path, 'w') as f:
        for label_filename in os.listdir(label_dir):
            truncation_start = label_filename.find('.bmp')
            img_filename = label_filename[:truncation_start] + label_filename[truncation_start + 28:]
            img_path = os.path.join(img_path_prefix, img_filename)
            label_path = os.path.join(label_path_prefix, label_filename)
            f.write(f'{img_path},{label_path}\n')

            if not os.path.exists(os.path.join(data_root, img_path)):
                print(img_filename)
            if not os.path.exists(os.path.join(data_root, label_path)):
                print(label_filename)

            cnt += 1

    print('共计:', cnt)


def create_index_file_ussd(img_dir, save_dir, train_rate):
    label_filenames = [filename for filename in os.listdir(img_dir) if filename.find('label') != -1]
    random.shuffle(label_filenames)
    train_num = int(len(label_filenames) * train_rate)
    print(f'训练集{train_num}, 测试集{len(label_filenames) - train_num}张')
    with open(os.path.join(save_dir, 'train.txt'), 'w') as train_file:
        train_set_labels = label_filenames[:train_num]
        for label_filename in train_set_labels:
            img_filename = label_filename.replace('_gray_label_', '_')
            img_filename = img_filename.replace('.png', '.jpg')
            train_file.write(f"{img_filename},{label_filename}\n")

    with open(os.path.join(save_dir, 'test.txt'), 'w') as test_file:
        test_set_labels = label_filenames[train_num:]
        for label_filename in test_set_labels:
            img_filename = label_filename.replace('_gray_label_', '_')
            img_filename = img_filename.replace('.png', '.jpg')
            test_file.write(f"{img_filename},{label_filename}\n")


if __name__ == '__main__':
    # create_index_file(data_root=r'/Datasets/Potsdam',
    #                   img_dir=r'/Datasets/Potsdam/train/img-png',
    #                   label_dir=r'/Datasets/Potsdam/train/label_gray',
    #                   save_path='../datasets/Potsdam/test.txt',
    #                   subset='train')

    create_index_file_ussd('/root/Datasets/USSD',
                           '/root/BiSeNet_v1_v2_PyTorch/datasets/USSD', 0.8)
