import os
import sys
import argparse
import torch
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

sys.path.insert(0, '.')
import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory


torch.set_grad_enabled(False)


def calculate():
    global_y_true = []
    global_y_pred = []

    to_tensor = T.ToTensor(
        mean=cfg.mean_value,
        std=cfg.std_value,
    )

    with open(cfg.val_im_anns) as f:
        pairs = f.read().splitlines()

    for pair in tqdm(pairs):
        img_path, label_path = pair.split(',')
        # img = cv2.imread(os.path.join(cfg.im_root, img_path), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imread(os.path.join(cfg.im_root, img_path))[:, :, ::-1]
        img = to_tensor(dict(im=img, lb=None))['im'].unsqueeze(0).cuda()
        label = cv2.imread(os.path.join(cfg.im_root, label_path), cv2.IMREAD_GRAYSCALE)

        out = net(img)[0].argmax(dim=1).squeeze().detach().cpu().numpy()

        global_y_true.append(label)
        global_y_pred.append(out)

    # global y_true, y_pred
    global_y_true = np.asarray(global_y_true).flatten()
    global_y_pred = np.asarray(global_y_pred).flatten()

    oa = accuracy_score(global_y_true, global_y_pred)
    kappa = cohen_kappa_score(global_y_true, global_y_pred)
    cm = confusion_matrix(global_y_true, global_y_pred)
    cr = classification_report(global_y_true, global_y_pred)

    print(oa)
    print(kappa)
    print(cm)
    print(cr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default='bisenetv2')
    parser.add_argument('--weight-path', type=str)
    args = parser.parse_args()
    cfg = cfg_factory[args.model]

    # define model
    net = model_factory[cfg.model_type](cfg.n_classes)
    # net.load_state_dict(torch.load(args.weight_path))
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.eval()
    net.cuda()

    calculate()

