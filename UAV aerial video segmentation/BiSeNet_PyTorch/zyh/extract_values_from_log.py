import argparse
import re


def extract_values_from_loss_log(log_path, result_save_path):
    re_data = re.compile(r'INFO logger\.py\(47\): iter: (\d+)/(\d+), lr: ([0-9.]+), eta: .*?,'
                         r' time: .*?, loss: ([0-9.]+), loss_pre: ([0-9.]+), loss_aux0: ([0-9.]+),'
                         r' loss_aux1: ([0-9.]+), loss_aux2: ([0-9.]+), loss_aux3: ([0-9.]+)')

    with open(log_path) as f:
        lines = f.read().split('\n')

    # cnt = 0
    csv = open(result_save_path, 'w')
    csv.write('iter, lr, loss, loss_pre, loss_aux0, loss_aux1, loss_aux2, loss_aux3\n')
    for line in lines:
        match = re_data.match(line)
        if match:
            data = match.groups()
            # cnt += 1
            csv.write(f'{data[0]}, {data[2]}, {data[3]}, {data[4]}, {data[5]}, {data[6]}, {data[7]}, {data[8]}\n')

    csv.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='Path of the log file')
    parser.add_argument('--save_path', type=str, help='Path of the result file')
    args = parser.parse_args()

    extract_values_from_loss_log(log_path=args.log_path, result_save_path=args.save_path)
