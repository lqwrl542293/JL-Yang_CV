import argparse
import os.path as ops
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--onnx_file_path',
        type=str,
        help='The converted onnx model file path',
    )
    parser.add_argument(
        '--output_trt_file_path',
        type=str,
        help='The output TensorRT engine file path',
    )

    return parser.parse_args()


def convert_onnx_into_tensorrt_engine(onnx_model_file_path, trt_engine_output_file):
    """

    :param onnx_model_file_path:
    :param trt_engine_output_file:
    :return:
    """
    if ops.exists(trt_engine_output_file):
        print('Trt engine file: {:s} has been generated'.format(trt_engine_output_file))
        return
    try:
        with trt.Builder(TRT_LOGGER) as builder:
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with builder.create_network(explicit_batch) as network:
                with trt.OnnxParser(network, TRT_LOGGER) as parser:
                    # Parse the model to create a network.
                    with open(onnx_model_file_path, 'rb') as model:
                        parser.parse(model.read())
                        for error_index in range(parser.num_errors):
                            print(parser.get_error(error_index).desc())
                            print(parser.get_error(error_index).code())
                            print(parser.get_error(error_index).file())

                    # Configure the builder here.
                    builder.max_batch_size = 8
                    builder.max_workspace_size = 1 << 32

                    # Build and return the engine. Note that the builder,
                    # network and parser are destroyed when this function returns.
                    engine = builder.build_cuda_engine(network)
                    if engine is not None:
                        with open(trt_engine_output_file, "wb") as f:
                            f.write(engine.serialize())
                        print('Successfully construct trt engine')
                        return engine
                    else:
                        print('Failed to construct trt engine')
                        return engine
    except Exception as err:
        print(err)
        print('Failed to construct trt engine')
        return None


if __name__ == '__main__':
    args = init_args()
    convert_onnx_into_tensorrt_engine(onnx_model_file_path=args.onnx_file_path,
                                      trt_engine_output_file=args.output_trt_file_path)
