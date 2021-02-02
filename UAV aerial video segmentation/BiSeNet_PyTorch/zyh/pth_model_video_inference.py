import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from configs import cfg_factory
from lib.models import model_factory
import lib.transform_cv2 as T

# COLOR_MAP = [(0, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128), (255, 0, 0)]
COLOR_MAP = [(0, 0, 0), (0, 255, 255), (255, 0, 0), (0, 255, 0), (128, 0, 128), (0, 0, 255)]


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='The model file path',
    )
    parser.add_argument(
        '--input_video_path',
        type=str,
        help='The input video file path'
    )
    parser.add_argument(
        '--save_path',
        type=str,
    )
    parser.add_argument('--model', type=str)
    return parser.parse_args()


def convert_gray_label_to_mask(label):
    mask = np.zeros(shape=[cfg.trt_engine_input_size[0], cfg.trt_engine_input_size[1], 3], dtype=np.uint8)
    unique_label_ids = np.unique(label)
    for label_id in unique_label_ids:
        idx = np.where(label == label_id)
        mask[idx] = COLOR_MAP[label_id]
    return mask


def create_input_output():
    cap = cv2.VideoCapture(args.input_video_path)

    width, height = int(cap.get(3)), int(cap.get(4))
    size = (width, height)  # TODO
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    out.open(args.save_path, fourcc, fps, size)

    return cap, out


def pth_video_inference(cap, out):
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:, :, (2, 1, 0)]
        frame = cv2.resize(frame, dsize=(cfg.trt_engine_input_size[1], cfg.trt_engine_input_size[0]),
                           interpolation=cv2.INTER_LINEAR)
        im = to_tensor(dict(im=frame, lb=None))['im'].unsqueeze(0).cuda()
        pred = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        mask = convert_gray_label_to_mask(pred)
        out.write(mask)

        cnt += 1
        cv2.imwrite(f'/root/BiSeNet_v1_v2_PyTorch/zyh/demos/frames/{cnt}.png', mask)

    cap.release()
    out.release()


def images_inference():
    with open(cfg.val_im_anns) as f:
        pairs = f.read().splitlines()

    cnt = 0
    for pair in tqdm(pairs):
        img_path, _ = pair.split(',')
        img = cv2.imread(os.path.join(cfg.im_root, img_path))[:, :, ::-1]
        img = to_tensor(dict(im=img, lb=None))['im'].unsqueeze(0).cuda()

        pred = net(img)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        pred = convert_gray_label_to_mask(pred)

        cnt += 1
        cv2.imwrite(os.path.join(args.save_path, f'{cnt}.png'), pred)


if __name__ == '__main__':
    args = init_args()
    cfg = cfg_factory[args.model]

    net = model_factory[cfg.model_type](cfg.n_classes)
    net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    net.eval()
    net.cuda()

    # prepare data
    to_tensor = T.ToTensor(
        mean=cfg.mean_value,
        std=cfg.std_value,
    )

    if args.mode == 'video':
        cap, out = create_input_output()
        pth_video_inference(cap, out)
    elif args.mode == 'test_set':
        images_inference()
