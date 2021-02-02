import os
import numpy as np
import cv2
from tqdm import tqdm


IMG_HEIGHT = 512
IMG_WIDTH = 512


def calculate_mean_std(img_dir):
    r_channel = 0
    g_channel = 0
    b_channel = 0

    filenames = os.listdir(img_dir)
    for filename in tqdm(filenames):
        img = cv2.imdecode(np.fromfile(os.path.join(img_dir, filename), np.uint8), cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = img / 255.0

        r_channel += np.sum(img[:, :, 2])
        g_channel += np.sum(img[:, :, 1])
        b_channel += np.sum(img[:, :, 0])

    n = len(filenames) * IMG_HEIGHT * IMG_WIDTH

    r_mean = r_channel / n
    g_mean = g_channel / n
    b_mean = b_channel / n

    print(r_mean, g_mean, b_mean)

    r_channel = 0
    g_channel = 0
    b_channel = 0
    for filename in tqdm(filenames):
        img = cv2.imdecode(np.fromfile(os.path.join(img_dir, filename), np.uint8), cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = img / 255.0

        r_channel += np.sum((img[:, :, 2] - r_mean) ** 2)
        g_channel += np.sum((img[:, :, 1] - g_mean) ** 2)
        b_channel += np.sum((img[:, :, 0] - b_mean) ** 2)

    r_std = np.sqrt(r_channel / n)
    g_std = np.sqrt(g_channel / n)
    b_std = np.sqrt(b_channel / n)

    print(r_std, g_std, b_std)


if __name__ == '__main__':
    calculate_mean_std(img_dir=r'F:\MachineLearning-Datasets\Potsdam\train\img-png')

# 0.336475088071354 0.3596638348772777 0.3330607973729019
# 0.14048183237065717 0.13784573260302724 0.14324438225871627
