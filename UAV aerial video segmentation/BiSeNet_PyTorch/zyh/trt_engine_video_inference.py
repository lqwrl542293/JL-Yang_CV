import os
import sys
import argparse
import time
import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 虽然没有直接用到但也不能去掉，pycuda.driver需要初始化，得到context

sys.path.insert(0, '.')
from configs import cfg_factory
from lib.models import model_factory
import lib.transform_cv2 as T

# COLOR_MAP = [(0, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128), (255, 0, 0)]
COLOR_MAP = [(0, 0, 0), (0, 255, 255), (255, 0, 0), (0, 255, 0), (128, 0, 128), (0, 0, 255)]
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        help='The TensorRT engine file path',
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
    cap = cv2.VideoCapture(args.input_video)

    width, height = int(cap.get(3)), int(cap.get(4))
    size = (width, height)  # TODO
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    out.open(args.save_path, fourcc, fps, size)

    return cap, out


def trt_engine_video_inference(cap, out):
    with open(args.model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    img_mean = np.array(cfg.mean_value).reshape((1, 1, len(cfg.mean_value)))
    img_std = np.array(cfg.std_value).reshape((1, 1, len(cfg.std_value)))

    # read input image file
    h_input = np.empty(shape=[cfg.trt_engine_input_size[0], cfg.trt_engine_input_size[1], 3], dtype=np.float32)
    h_output = np.empty(shape=[cfg.trt_engine_input_size[0], cfg.trt_engine_input_size[1]], dtype=np.int32)

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * h_input.nbytes)
    d_output = cuda.mem_alloc(1 * h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    cnt = 0
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        t_start = time.time()

        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break

            frame = frame[:, :, (2, 1, 0)]
            frame = cv2.resize(frame, dsize=(cfg.trt_engine_input_size[1], cfg.trt_engine_input_size[0]),
                               interpolation=cv2.INTER_LINEAR)
            frame = frame.astype('float32') / 255.0
            frame -= img_mean
            frame /= img_std

            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, frame, stream)
            # Run inference.
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(h_output, d_output, stream)

            h_output = h_output.astype(np.uint8)
            mask = convert_gray_label_to_mask(h_output)
            out.write(mask)

            cnt += 1
            cv2.imwrite(f'/root/BiSeNet_v1_v2_PyTorch/zyh/demos/frames/{cnt}.png', mask)

        cost_time = time.time() - t_start
        # Synchronize the stream
        stream.synchronize()

        print('Total inference cost: {:.5f}'.format(cost_time))
        time_per_frame = cost_time / cnt
        print('Inference cost per frame: {:.5f}'.format(time_per_frame))
        print('Inference fps: {:.5f}'.format(1 / time_per_frame))

        cap.release()
        out.release()


if __name__ == '__main__':
    args = init_args()
    cfg = cfg_factory[args.model]

    cap, out = create_input_output()
    trt_engine_video_inference(cap, out)
