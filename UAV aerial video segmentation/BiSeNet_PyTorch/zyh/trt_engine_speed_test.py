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


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trt_file_path',
        type=str,
        help='The TensorRT engine file path',
    )
    parser.add_argument(
        '--input_image_path',
        type=str,
        help='The input testing image file path'
    )
    parser.add_argument('--model', type=str)
    return parser.parse_args()


def time_profile_trt_engine(image_file_path, trt_engine_file_path):
    with open(trt_engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # read input image file
    h_input = np.empty(shape=[cfg.trt_engine_input_size[1], cfg.trt_engine_input_size[0], 3], dtype=np.float32)
    h_output = np.empty(shape=[cfg.trt_engine_input_size[1], cfg.trt_engine_input_size[0]], dtype=np.int32)

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * h_input.nbytes)
    d_output = cuda.mem_alloc(1 * h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # read images
    src_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    src_image = src_image[:, :, (2, 1, 0)]
    src_image = cv2.resize(src_image, dsize=(cfg.trt_engine_input_size[1], cfg.trt_engine_input_size[0]),
                           interpolation=cv2.INTER_LINEAR)
    src_image = src_image.astype('float32') / 255.0
    img_mean = np.array(cfg.mean_value).reshape((1, 1, len(cfg.mean_value)))
    img_std = np.array(cfg.std_value).reshape((1, 1, len(cfg.std_value)))
    src_image -= img_mean
    src_image /= img_std

    loop_times = 5000

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        t_start = time.time()
        for i in range(loop_times):
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, src_image, stream)
            # Run inference.
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
        cost_time = (time.time() - t_start) / loop_times
        # Synchronize the stream
        stream.synchronize()
        print('Inference cost: {:.5f}'.format(cost_time))
        print('Inference fps: {:.5f}'.format(1 / cost_time))


if __name__ == '__main__':
    args = init_args()
    cfg = cfg_factory[args.model]

    # 注意图像由cv2读入，按BGR的顺序
    # MEAN_VALUE = (0.3331, 0.3597, 0.3365)
    # STD_VALUE = (0.1432, 0.1378, 0.1405)

    time_profile_trt_engine(image_file_path=args.input_image_path,
                            trt_engine_file_path=args.trt_file_path)
